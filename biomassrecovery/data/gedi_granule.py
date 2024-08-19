"""Module for convenient objects to deal with GEDI data products"""
from __future__ import annotations

import datetime
import pathlib
import re
from dataclasses import dataclass
from typing import Iterable, Union, List

import geopandas as gpd
import geopandas.array
import h5py
import numpy as np
import pandas as pd
from shapely.geometry import box
import xarray

from biomassrecovery.constants import WGS84


@dataclass
class GediNameMetadata:
    """Data class container for metadata derived from GEDI file name conventions."""

    product: str
    year: str
    julian_day: str
    hour: str
    minute: str
    second: str
    orbit: str
    ground_track: str
    positioning: str
    granule_production_version: str
    release_number: str


@dataclass
class GediLPNameMetadata(GediNameMetadata):
    """Data class container for metadata derived from GEDI file names
    released by the LP DAAC"""

    sub_orbit_granule: str
    pge_version_number: str


@dataclass
class GediORNLNameMetadata(GediNameMetadata):
    """Data class container for metadata derived from GEDI file names
    released by the ORNL DAAC"""


GEDI_SUBPATTERN_LP = GediLPNameMetadata(
    product=r"\w+_\w",
    year=r"\d{4}",
    julian_day=r"\d{3}",
    hour=r"\d{2}",
    minute=r"\d{2}",
    second=r"\d{2}",
    orbit=r"O\d+",
    sub_orbit_granule=r"\d{2}",
    ground_track=r"T\d+",
    positioning=r"\d{2}",
    pge_version_number=r"\d{3}",
    granule_production_version=r"\d{2}",
    release_number=r"V\d+",
)

GEDI_SUBPATTERN_ORNL = GediORNLNameMetadata(
    product=r"\w+_\w",
    year=r"\d{4}",
    julian_day=r"\d{3}",
    hour=r"\d{2}",
    minute=r"\d{2}",
    second=r"\d{2}",
    orbit=r"O\d+",
    ground_track=r"T\d+",
    positioning=r"\d{2}",
    release_number=r"\d{3}",
    granule_production_version=r"\d{2}",
)


def _parse_lp_granule_filename(gedi_filename: str) -> GediLPNameMetadata:
    GEDI_SUBPATTERN = GEDI_SUBPATTERN_LP
    gedi_naming_pattern = re.compile(
        (
            f"({GEDI_SUBPATTERN.product})"
            f"_({GEDI_SUBPATTERN.year})"
            f"({GEDI_SUBPATTERN.julian_day})"
            f"({GEDI_SUBPATTERN.hour})"
            f"({GEDI_SUBPATTERN.minute})"
            f"({GEDI_SUBPATTERN.second})"
            f"_({GEDI_SUBPATTERN.orbit})"
            f"_({GEDI_SUBPATTERN.sub_orbit_granule})"
            f"_({GEDI_SUBPATTERN.ground_track})"
            f"_({GEDI_SUBPATTERN.positioning})"
            f"_({GEDI_SUBPATTERN.pge_version_number})"
            f"_({GEDI_SUBPATTERN.granule_production_version})"
            f"_({GEDI_SUBPATTERN.release_number})"
        )
    )
    parse_result = re.search(gedi_naming_pattern, gedi_filename)
    if parse_result is None:
        raise ValueError(
            f"Filename {gedi_filename} does not conform the the GEDI naming pattern."
        )
    return GediLPNameMetadata(*parse_result.groups())


def _parse_ornl_granule_filename(gedi_filename: str) -> GediORNLNameMetadata:
    GEDI_SUBPATTERN = GEDI_SUBPATTERN_ORNL
    gedi_naming_pattern = re.compile(
        (
            f"({GEDI_SUBPATTERN.product})"
            f"_({GEDI_SUBPATTERN.year})"
            f"({GEDI_SUBPATTERN.julian_day})"
            f"({GEDI_SUBPATTERN.hour})"
            f"({GEDI_SUBPATTERN.minute})"
            f"({GEDI_SUBPATTERN.second})"
            f"_({GEDI_SUBPATTERN.orbit})"
            f"_({GEDI_SUBPATTERN.ground_track})"
            f"_({GEDI_SUBPATTERN.positioning})"
            f"_({GEDI_SUBPATTERN.release_number})"
            f"_({GEDI_SUBPATTERN.granule_production_version})"
        )
    )
    parse_result = re.search(gedi_naming_pattern, gedi_filename)
    if parse_result is None:
        raise ValueError(
            f"Filename {gedi_filename} does not conform the the GEDI naming pattern."
        )
    return GediORNLNameMetadata(*parse_result.groups())


def _parse_gedi_granule_filename(gedi_filename: str) -> GediNameMetadata:
    """
    Parse a GEDI granule filename for the relevant metadata contained in the name.

    Args:
        gedi_filename (str): The filename to parse.

    Raises:
        ValueError: If the filename gedi_filename does not follow the GEDI conventions

    Returns:
        GediNameMetadata: The parsed metadata in a dataclass container
    """

    # The LP DAAC and ORNL DAAC use slightly different naming conventions.
    # Check which is being used here
    gedi_product_pattern = re.compile(r"GEDI([0-9]+)_(\w)")
    parse_result = re.search(gedi_product_pattern, gedi_filename)
    if parse_result is None:
        raise ValueError(
            f"Filename {gedi_filename} does not conform the the GEDI naming pattern."
        )
    level = int(parse_result.group(1))
    # Levels 1 and 2 are distributed by the LP DAAC
    if level == 1 or level == 2:
        return _parse_lp_granule_filename(gedi_filename)
    # Levels 3 and 4 are distributed by the ORNL DAAC
    elif level == 3 or level == 4:
        return _parse_ornl_granule_filename(gedi_filename)
    else:
        raise ValueError(
            f"Filename {gedi_filename} does not conform the the GEDI naming pattern."
        )


class GediGranule(h5py.File):  # TODO  pylint: disable=missing-class-docstring
    def __init__(self, file_path: pathlib.Path):
        super().__init__(file_path, "r")
        self.file_path = file_path
        self.beam_names = [
            name for name in self.keys() if name.startswith("BEAM")
        ]
        self._parsed_filename_metadata = None

    @property
    def version(self) -> str:
        return self["METADATA"]["DatasetIdentification"].attrs["VersionID"]

    @property
    def filename_metadata(self) -> GediNameMetadata:
        if self._parsed_filename_metadata is None:
            self._parsed_filename_metadata = _parse_gedi_granule_filename(
                self.filename
            )
        return self._parsed_filename_metadata

    @property
    def start_datetime(self) -> pd.Timestamp:
        return pd.to_datetime(
            (
                f"{self.filename_metadata.year}"
                f".{self.filename_metadata.julian_day}"
                f".{self.filename_metadata.hour}"
                f":{self.filename_metadata.minute}"
                f":{self.filename_metadata.second}"
            ),
            format="%Y.%j.%H:%M:%S",
        )

    @property
    def product(self) -> str:
        return self["METADATA"]["DatasetIdentification"].attrs["shortName"]

    @property
    def uuid(self) -> str:
        return self["METADATA"]["DatasetIdentification"].attrs["uuid"]

    @property
    def filename(self) -> str:
        return self["METADATA"]["DatasetIdentification"].attrs["fileName"]

    @property
    def abstract(self) -> str:
        return self["METADATA"]["DatasetIdentification"].attrs["abstract"]

    @property
    def n_beams(self) -> int:
        return len(self.beam_names)

    def beam(self, identifier: Union[str, int]) -> GediBeam:

        if isinstance(identifier, int):
            return self._beam_from_index(identifier)
        elif isinstance(identifier, str):
            return self._beam_from_name(identifier)
        else:
            raise ValueError(
                "identifier must either be the beam index or beam name"
            )

    def _beam_from_index(self, beam_index: int) -> GediBeam:
        if not 0 <= beam_index < self.n_beams:
            raise ValueError(
                f"Beam index must be between 0 and {self.n_beams-1}"
            )

        beam_name = self.beam_names[beam_index]
        return self._beam_from_name(beam_name)

    def _beam_from_name(self, beam_name: str) -> GediBeam:
        if not beam_name in self.beam_names:
            raise ValueError(f"Beam name must be one of: {self.beam_names}")
        return GediBeam(granule=self, beam_name=beam_name)

    def iter_beams(self) -> Iterable[GediBeam]:
        for beam_index in range(self.n_beams):
            yield self._beam_from_index(beam_index)

    def list_beams(self) -> list[GediBeam]:
        return list(self.iter_beams())

    def close(self) -> None:
        super().close()

    def __repr__(self) -> str:
        try:
            description = (
                "GEDI Granule:\n"
                f" Granule name: {self.filename}\n"
                f" Sub-granule:  {self.filename_metadata.sub_orbit_granule}\n"
                f" Product:      {self.product}\n"
                f" Release:      {self.filename_metadata.release_number}\n"
                f" No. beams:    {self.n_beams}\n"
                f" Start date:   {self.start_datetime.date()}\n"
                f" Start time:   {self.start_datetime.time()}\n"
                f" HDF object:   {super().__repr__()}"
            )
        except AttributeError:
            description = (
                "GEDI Granule:\n"
                f" Granule name: {self.filename}\n"
                f" Product:      {self.product}\n"
                f" No. beams:    {self.n_beams}\n"
                f" Start date:   {self.start_datetime.date()}\n"
                f" Start time:   {self.start_datetime.time()}\n"
                f" HDF object:   {super().__repr__()}"
            )
        return description


class GediBeam(h5py.Group):
    """
    Class containing GEDI data for a single beam of a granule.

    Args:
        granule: The parent granule for this beam
        beam name: The name of this beam, e.g. BEAM0000
        roi (optional): A bounding box for the region of interest.
            GEDI granule files include the entire global granule track,
            which is often not needed for a small region of interest. If
            a bounding box is provided, this class will *attempt* to perform
            optimizations such that computationally intensive data parsing
            operations take place only on the data within the region of interest.
            This will result in ONLY shots falling within the roi being included
            in the cached data (returned by main_data
            To re-parse the file with a new roi, use beam.reset_roi().
    """

    def __init__(self, granule: GediGranule, beam_name: str, roi: box = None):
        super().__init__(granule[beam_name].id)
        self.parent_granule = granule  # Reference to parent granule
        self._cached_data = None
        self._shot_geolocations = None
        self.roi = roi

    def list_datasets(self, top_level_only: bool = True) -> list[str]:
        if top_level_only:
            return list(self)
        else:
            # TODO
            raise NotImplementedError

    @property
    def name(self) -> str:
        return super().name[1:]

    @property
    def beam_type(self) -> str:
        return self.attrs["description"].split(" ")[0].lower()

    @property
    def quality(self) -> h5py.Dataset:
        return self["quality_flag"]

    @property
    def sensitivity(self) -> h5py.Dataset:
        return self["sensitivity"]

    @property
    def geolocation(self) -> h5py.Dataset:
        return self["geolocation"]

    @property
    def n_shots(self) -> int:
        return len(self["beam"])

    @property
    def shot_geolocations(self) -> geopandas.array.GeometryArray:
        """
        Return an array of shapely Point objects at the (lon, lat) of each shot.

        Note:
        For GEDI_L1B products the (lon, lat) coordinates of the last bin in the return
        waveform are returned.
        For GEDI_L2A products the (lon, lat) coordinates of the lowest detected mode
        (i.e. ground mode), as detected by the GEDI selected algorithm, are returned.
        As a result, it is expected that the (lon, lat) values for the same shot but
        different products is not exactly the same.

        Returns:
            geopandas.array.GeometryArray: A geometry array containing shapely Point
                objects at the (longitude, latitude) positions of the shots in the beam.
                Longitude, Latitude coordinates are given in the WGS84 coordinate
                reference system.
        """
        if self._shot_geolocations is None:
            if self.parent_granule.product == "GEDI_L4A":
                self._shot_geolocations = gpd.points_from_xy(
                    x=self["lon_lowestmode"],
                    y=self["lat_lowestmode"],
                    crs=WGS84,
                )
            elif self.parent_granule.product == "GEDI_L2A":
                self._shot_geolocations = gpd.points_from_xy(
                    x=self["lon_lowestmode"],
                    y=self["lat_lowestmode"],
                    crs=WGS84,
                )
            elif self.parent_granule.product == "GEDI_L2B":
                self._shot_geolocations = gpd.points_from_xy(
                    x=self["geolocation/lon_lowestmode"],
                    y=self["geolocation/lat_lowestmode"],
                    crs=WGS84,
                )
            elif self.parent_granule.product == "GEDI_L1B":
                self._shot_geolocations = gpd.points_from_xy(
                    x=self["geolocation/longitude_lastbin"],
                    y=self["geolocation/latitude_lastbin"],
                    crs=WGS84,
                )
            else:
                raise NotImplementedError(
                    "No method to get main data for "
                    f"product {self.parent_granule.product}"
                )
        return self._shot_geolocations

    @property
    def main_data(self, sql_format_arrays=False) -> gpd.GeoDataFrame:
        """
        Return the main data for all shots in beam as geopandas DataFrame.

        Supports the following products: GEDI_L1B, GEDI_L2A

        Returns:
            gpd.GeoDataFrame: A geopandas DataFrame containing the main data for the given beam object.
        """
        if self._cached_data is None:
            data = self._get_main_data_dict()
            geometry = self.shot_geolocations
            self._cached_data = gpd.GeoDataFrame(
                data, geometry=geometry, crs=WGS84
            )

        return self._cached_data

    def sql_format_arrays(self) -> None:
        """Forces array-type fields to be sql-formatted (text strings).

        Until this function is called, array-type fields will be np.array() objects. This formatting can be undone by resetting the cache."""
        array_cols = [c for c in self.main_data.columns if c.endswith("_z")]
        for c in array_cols:
            self._cached_data[c] = self.main_data[c].map(self._arr_to_str)

    def reset_cache(self):
        self._cached_data = None

    def reset_roi(self, new_roi: box = None) -> None:
        if new_roi == self.roi:
            return None
        self.roi = new_roi
        self._cached_data = None

    def _get_main_data_dict(self) -> dict:
        """Returns correct main data depending on product"""
        if self.parent_granule.product == "GEDI_L1B":
            return self._get_gedi1b_main_data_dict()
        elif self.parent_granule.product == "GEDI_L2A":
            return self._get_gedi2a_main_data_dict()
        elif self.parent_granule.product == "GEDI_L2B":
            return self._get_gedi2b_main_data_dict()
        elif self.parent_granule.product == "GEDI_L4A":
            return self._get_gedi4a_main_data_dict()
        else:
            raise NotImplementedError(
                f"No method to get main data for product {self.parent_granule.product}"
            )

    def _accumulate_waveform_data(
        self, name: str, start: int, end: int
    ) -> np.array:
        waveform_data_all = np.array(self[name][:])
        data = []
        for i in range(len(start)):
            dz = waveform_data_all[
                start[i] : end[i]
            ]  # this is a view, not a copy
            data.append(dz)
        return data

    def _arr_to_str(self, arr: Union[List[float], np.array]) -> str:
        """Converts array type data to SQL-friendly string."""
        return "{" + ", ".join(map(str, arr)) + "}"

    def _get_gedi4a_main_data_dict(self) -> dict:
        """
        Return the main data for all shots in a GEDI L4A product beam as a dictionary.
        Download the L4A data dictionary from
        https://daac.ornl.gov/GEDI/guides/GEDI_L4A_AGB_Density.html for details
        of all the available variables.

        Returns: A dictionary containing the main data for all shots in the given
            beam of the granule.
        """
        gedi_l4a_count_start = pd.to_datetime("2018-01-01T00:00:00Z")
        # # Not needed as acillary data same for all granules
        # predict_stratum = self["predict_stratum"][:]
        # ancillary_stratum = self["ANCILLARY/model_data"]['predict_stratum']

        # # extracting the corresponding ancillary_data index for every shot
        # ancillary_idx = np.zeros(predict_stratum.shape, dtype=int)
        # for i, byte in enumerate(predict_stratum):
        #     # Get the index of the byte in the reference array
        #     index = np.where(ancillary_stratum == byte)[0]
            
        #     # If the byte is found, take the first occurrence index
        #     if index.size > 0:
        #         ancillary_idx[i] = index[0]
        #     else:
        #         # If not found, raise error
        #         raise ValueError("No matching predict_stratum value found for a shot {} in beam {} in granule {}".format(self["shot_number"][:][i],
        #                                                                                                                  self.name,
                                                                                                                        #  self.parent_granule.filename)
                                                                                                                        #  )
        
        # for more details compare: https://daac.ornl.gov/GEDI/guides/GEDI_L4A_AGB_Density.html and https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022EA002516
        data = {
            # General identifiable data
            "shot_number": self["shot_number"][:],
            "granule_name": [self.parent_granule.filename] * self.n_shots,
            # channel Channel identifier
            # beam Beam identifier
            "beam_name": [self.name] * self.n_shots,
            "beam_type": [self.beam_type] * self.n_shots,

            # Temporal data
            "delta_time": self["delta_time"][:], # Time delta since Jan 1 00:00 2018
            "absolute_time": (
                gedi_l4a_count_start
                + pd.to_timedelta(self["delta_time"], unit="seconds")
                ),
            
            # Quality data
            "sensitivity": self["sensitivity"][:], # Maximum canopy cover that can be penetrated considering the SNR of the waveform
            "degrade_flag": self["degrade_flag"][:], # Non-zero values indicate the shot occurred during a degraded period of pointing and/or positioning information
            # A non-zero tens digit indicates degraded attitude, a non-zero units digit indicates a degraded trajectory. Details: p.18, https://lpdaac.usgs.gov/documents/998/GEDI02_UserGuide_V21.pdf
            # We only want to save GEDI shots with reliable biomass predictions (l4_quality_flag == 1) to the DB
            # l4_quality_flag == 1 if: l2_quality_flag == 1, sensitivity > 0.95, landsat_water_persistence < 10, leaf_off_flag = 0 for decideous trees,  urban_proportion < 50
            # l2_quality_flag == 1 if: algorithm_run_flag = 1, surface_flag = 1, stale_return_flag = 0, sensitivity > 0.9, rx_maxamp > 8 × sd_corrected 
            "l4_quality_flag": self["l4_quality_flag"][:], # Flag simplifying selection of most useful biomass predictions
            "predictor_limit_flag": self["predictor_limit_flag"][:], # RH predictor value is (0=in bounds; 1=outside lower bound; 2=outside upper bound) of training data
            "response_limit_flag": self["response_limit_flag"][:], # AGBD predicton value is (0=in bounds; 1=outside lower bound; 2=outside upper bound) of training data
            # solar_elevation # Solar elevation angle

            # Processing data
            "selected_algorithm": self["selected_algorithm"][:],
            "selected_mode": self["selected_mode"][:],  # ID of mode selected as lowest non-noise mode

            # Geolocation data
            "lat_lowestmode": self["lat_lowestmode"][:], # Latitude of center of lowest mode
            "lon_lowestmode": self["lon_lowestmode"][:], # Longitude of center of lowest mode
            "elev_lowestmode": self["elev_lowestmode"][:], # Elevation of center of lowest mode relative to reference ellipsoid
            
            # ABGD data
            "agbd": self["agbd"][:], # Aboveground biomass density (Mg/ha)
            "agbd_pi_lower": self["agbd_pi_lower"][:], # prediction intervals for alpha = 0.05
            "agbd_pi_upper": self["agbd_pi_upper"][:],
            "agbd_se": self["agbd_se"][:],
            "agbd_t": self["agbd_t"][:], # Model prediction in fit units
            "agbd_t_se": self["agbd_t_se"][:],

            # Land cover data
            "predict_stratum": self["predict_stratum"][:], # stratum information predicted by non-GEDI data for the 1 km cell (Plant Functional Type (PFT) derived from the MODIS MCD12Q1 V006 product)
            "landsat_treecover": self["land_cover_data/landsat_treecover"][:],
            "urban_focal_window_size": self["land_cover_data/urban_focal_window_size"][:], # Window size used to calculate urban_proportion (3=3x3, 5=5x5 pixel window size).
            "urban_proportion": self["land_cover_data/urban_proportion"][:], # derived from the DLR 12 m resolution TanDEM-X Global Urban Footprint Product.
            # landsat_water_persistence: The percent UMD GLAD Landsat observations with classified surface water between 2018 and 2019. Values > 80 usually represent permanent water while values < 10 represent permanent land
            # BEAMXXXX/land_cover_data/pft_class and BEAMXXXX/land_cover_data/region_class are contained in BEAMXXXX/predict_stratum
            # "pft_class": self["land_cover_data/pft_class"][:],
            # "region_class": self["land_cover_data/region_class"][:],
            # landsat_treecover: Tree cover in the year 2010, defined as canopy closure for all vegetation taller than 5 m in height (Hansen et al., 2013) and encoded as a percentage per output grid cell
           
            # Ancillary
            # # Ancillary table is the same for all granules in whole dataset!
            # predict_stratum: compare landcover, in ancillary we get the matching ancillary data to each predict_stratum
            # so for each data point the specific predict_stratum is the index for the correct ancillary info
            # fit_stratum # where the data is from that was used to train the model used for stratum referred to in predict_stratum
            # model_name # the model used for stratum referred to in predict_stratum
            # "dof": self.parent_granule["ANCILLARY/model_data"]["dof"][ancillary_idx],
            # "y_transform": self.parent_granule["ANCILLARY/model_data"]["y_transform"][ancillary_idx],
            # "bias_correction_name": self.parent_granule["ANCILLARY"]["model_data/bias_correction_name"][ancillary_idx],
            # "bias_correction_value": self.parent_granule["ANCILLARY"]["model_data/bias_correction_value"][ancillary_idx],
            # # rse:  residual standard error from the linear regression applied to prediction stratum
        }
        return data

    def _get_gedi2b_main_data_dict(self) -> dict:
        """
        Return the main data for all shots in a GEDI L2B product beam as dictionary.

        Returns:
            dict: A dictionary containing the main data for all shots in the given
                beam of the granule.
        """
        gedi_l2b_count_start = pd.to_datetime("2018-01-01T00:00:00Z")
        data = {
            # General identifiable data
            "granule_name": [self.parent_granule.filename] * self.n_shots,
            "shot_number": self["shot_number"][:],
            "beam_type": [self.beam_type] * self.n_shots,
            "beam_name": [self.name] * self.n_shots,
            # Temporal data
            "delta_time": self["geolocation/delta_time"][:],
            "absolute_time": (
                gedi_l2b_count_start
                + pd.to_timedelta(self["delta_time"], unit="seconds")
            ),
            # Quality data
            "algorithm_run_flag": self["algorithmrun_flag"][:],
            "l2a_quality_flag": self["l2a_quality_flag"][:],
            "l2b_quality_flag": self["l2b_quality_flag"][:],
            "sensitivity": self["sensitivity"][:],
            "degrade_flag": self["geolocation/degrade_flag"][:],
            "stale_return_flag": self["stale_return_flag"][:],
            "surface_flag": self["surface_flag"][:],
            "solar_elevation": self["geolocation/solar_elevation"][:],
            "solar_azimuth": self["geolocation/solar_azimuth"][:],
            # Scientific data
            "cover": self["cover"][:],
            "cover_z": list(self["cover_z"][:]),
            "fhd_normal": self["fhd_normal"][:],
            "num_detectedmodes": self["num_detectedmodes"][:],
            "omega": self["omega"][:],
            "pai": self["pai"][:],
            "pai_z": list(self["pai_z"][:]),
            "pavd_z": list(self["pavd_z"][:].tolist()),
            "pgap_theta": self["pgap_theta"][:],
            "pgap_theta_error": self["pgap_theta_error"][:],
            "rg": self["rg"][:],
            "rh100": self["rh100"][:],
            "rhog": self["rhog"][:],
            "rhog_error": self["rhog_error"][:],
            "rhov": self["rhov"][:],
            "rhov_error": self["rhov_error"][:],
            "rossg": self["rossg"][:],
            "rv": self["rv"][:],
            "rx_range_highestreturn": self["rx_range_highestreturn"][:],
            # DEM
            "dem_tandemx": self["geolocation/digital_elevation_model"][:],
            # Land cover data: NOTE this is gridded and/or derived data
            "gridded_leaf_off_flag": self["land_cover_data/leaf_off_flag"][:],
            "gridded_leaf_on_doy": self["land_cover_data/leaf_on_doy"][:],
            "gridded_leaf_on_cycle": self["land_cover_data/leaf_on_cycle"][:],
            "interpolated_modis_nonvegetated": self[
                "land_cover_data/modis_nonvegetated"
            ][:],
            "interpolated_modis_treecover": self[
                "land_cover_data/modis_treecover"
            ][:],
            "gridded_pft_class": self["land_cover_data/pft_class"][:],
            "gridded_region_class": self["land_cover_data/region_class"][:],
            # Processing data
            "selected_l2a_algorithm": self["selected_l2a_algorithm"][:],
            "selected_rg_algorithm": self["selected_rg_algorithm"][:],
            # Geolocation data
            "lon_highestreturn": self["geolocation/lon_highestreturn"][:],
            "lon_lowestmode": self["geolocation/lon_lowestmode"][:],
            "longitude_bin0": self["geolocation/longitude_bin0"][:],
            "longitude_bin0_error": self["geolocation/longitude_bin0_error"][:],
            "lat_highestreturn": self["geolocation/lat_highestreturn"][:],
            "lat_lowestmode": self["geolocation/lat_lowestmode"][:],
            "latitude_bin0": self["geolocation/latitude_bin0"][:],
            "latitude_bin0_error": self["geolocation/latitude_bin0_error"][:],
            "elev_highestreturn": self["geolocation/elev_highestreturn"][:],
            "elev_lowestmode": self["geolocation/elev_lowestmode"][:],
            "elevation_bin0": self["geolocation/elevation_bin0"][:],
            "elevation_bin0_error": self["geolocation/elevation_bin0_error"][:],
            # waveform data
            "waveform_count": self["rx_sample_count"][:],
            "waveform_start": self["rx_sample_start_index"][:] - 1,
        }

        # handle array data
        ## could delete waveform start/count after storing waveform chunks
        start = data["waveform_start"]
        end = start + data["waveform_count"]
        data["pgap_theta_z"] = self._accumulate_waveform_data(
            "pgap_theta_z", start, end
        )
        return data

    def _get_gedi2a_main_data_dict(self) -> dict:
        """
        Return the main data for all shots in a GEDI L2A product beam as dictionary.

        Returns:
            dict: A dictionary containing the main data for all shots in the given
                beam of the granule.
        """
        gedi_l2a_count_start = pd.to_datetime("2018-01-01T00:00:00Z")
        data = {
            # General identifiable data
            "granule_name": [self.parent_granule.filename] * self.n_shots,
            "shot_number": self["shot_number"][:],
            "beam_type": [self.beam_type] * self.n_shots,
            "beam_name": [self.name] * self.n_shots,
            # Temporal data
            "delta_time": self["delta_time"][:],
            "absolute_time": (
                gedi_l2a_count_start
                + pd.to_timedelta(self["delta_time"], unit="seconds")
            ),
            # Quality data
            "sensitivity": self["sensitivity"][:],
            "quality_flag": self["quality_flag"][:],
            "solar_elevation": self["solar_elevation"][:],
            "solar_azimuth": self["solar_elevation"][:],
            "energy_total": self["energy_total"][:],
            # DEM
            "dem_tandemx": self["digital_elevation_model"][:],
            "dem_srtm": self["digital_elevation_model_srtm"][:],
            # Processing data
            "selected_algorithm": self["selected_algorithm"][:],
            "selected_mode": self["selected_mode"][:],
            # Geolocation data
            "lon_lowestmode": self["lon_lowestmode"][:],
            "longitude_bin0_error": self["longitude_bin0_error"][:],
            "lat_lowestmode": self["lat_lowestmode"][:],
            "latitude_bin0_error": self["latitude_bin0_error"][:],
            "elev_lowestmode": self["elev_lowestmode"][:],
            "elevation_bin0_error": self["elevation_bin0_error"][:],
            "lon_highestreturn": self["lon_highestreturn"][:],
            "lat_highestreturn": self["lat_highestreturn"][:],
            "elev_highestreturn": self["elev_highestreturn"][:],
        } | {f"rh{i}": self["rh"][:, i] for i in range(101)}
        return data

    def _get_gedi1b_main_data_dict(self) -> dict:
        """
        Return the main data for all shots in a GEDI L1B product beam as dictionary.

        Returns:
            dict: A dictionary containing the main data for all shots in the given
                beam of the granule.
        """
        data = {
            # General identifiable data
            "granule_name": [self.parent_granule.filename] * self.n_shots,
            "shot_number": self["shot_number"][:],
            "beam_type": [self.beam_type] * self.n_shots,
            "beam_name": [self.name] * self.n_shots,
            # Temporal data
            "delta_time": self["delta_time"][:],
            # Quality data
            "degrade": self["geolocation/degrade"][:],
            "stale_return_flag": self["stale_return_flag"][:],
            "solar_elevation": self["geolocation/solar_elevation"][:],
            "solar_azimuth": self["geolocation/solar_elevation"][:],
            "rx_energy": self["rx_energy"][:],
            # DEM
            "dem_tandemx": self["geolocation/digital_elevation_model"][:],
            "dem_srtm": self["geolocation/digital_elevation_model_srtm"][:],
            # geolocation bin0
            "latitude_bin0": self["geolocation/latitude_bin0"][:],
            "latitude_bin0_error": self["geolocation/latitude_bin0_error"][:],
            "longitude_bin0": self["geolocation/longitude_bin0"][:],
            "longitude_bin0_error": self["geolocation/longitude_bin0_error"][:],
            "elevation_bin0": self["geolocation/elevation_bin0"][:],
            "elevation_bin0_error": self["geolocation/elevation_bin0_error"][:],
            # geolocation lastbin
            "latitude_lastbin": self["geolocation/latitude_lastbin"][:],
            "latitude_lastbin_error": self[
                "geolocation/latitude_lastbin_error"
            ][:],
            "longitude_lastbin": self["geolocation/longitude_lastbin"][:],
            "longitude_lastbin_error": self[
                "geolocation/longitude_lastbin_error"
            ][:],
            "elevation_lastbin": self["geolocation/elevation_lastbin"][:],
            "elevation_lastbin_error": self[
                "geolocation/elevation_lastbin_error"
            ][:],
            # relative waveform position info in beam and ssub-granule
            "waveform_start": self["rx_sample_start_index"][:] - 1,
            "waveform_count": self["rx_sample_count"][:],
        }
        return data

    def __repr__(self) -> str:
        description = (
            "GEDI Beam object:\n"
            f" Beam name:  {self.name}\n"
            f" Beam type:  {self.attrs['description']}\n"
            f" Shots:      {self.n_shots}\n"
            f" HDF object: {super().__repr__()}"
        )
        return description

    def intersect(self, geometry):
        raise NotImplementedError

    @property
    def waveform(self):
        if self.parent_granule.product != "GEDI_L1B":
            raise NotImplementedError(
                "Waveforms only exist for GEDI_L1B products. "
                f"Current beam is from a {self.parent_granule.product} product."
            )

        return xarray.DataArray(
            self["rxwaveform"][:],
            dims=["sample_points"],
            name=f"{self.parent_granule.filename[:-3]}_{self.name}",
            attrs={
                "granule": self.parent_granule.filename,
                "beam": self.name,
                "type": self.beam_type,
                "creation_timestamp_utc": str(datetime.datetime.utcnow()),
            },
        )

    def save_waveform(
        self, save_dir: pathlib.Path, overwrite: bool = False
    ) -> None:
        waveform = self.waveform
        save_name = f"{self.parent_granule.filename[:-3]}_{self.name}.nc"
        save_path = pathlib.Path(save_dir) / save_name
        if overwrite or not save_path.exists():
            waveform.to_netcdf(save_path)
