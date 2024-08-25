#! /home/ah2174/biomass-recovery/venv/bin/python
import os
import pathlib
import numpy as np
from pyspark.sql import SparkSession


import argparse
import pandas as pd

from geopandas import gpd
import subprocess
from shapely.geometry import box
from shapely.geometry.polygon import orient
import tempfile
import shutil
import sqlalchemy

from biomassrecovery.data.gedi_cmr_query import query
from biomassrecovery.data.gedi_download_pipeline import check_and_format_shape, DetailError
from biomassrecovery.data import gedi_database_loader
from biomassrecovery import constants
from biomassrecovery import environment
from functools import partial
from typing import List, Optional
from biomassrecovery.utils import logging_util

logger = logging_util.get_logger(__name__)


def _get_engine():
    # Since spark runs workers in their own process, we cannot share database connections
    # between workers. We just create a new connection for each query. This is reasonable because
    # most of our queries involve inserting a large amount of data into the database.
    return sqlalchemy.create_engine(environment.DB_CONFIG, echo=False)


def _fetch_cookies():
    print("No authentication cookies found, fetching earthdata cookies ...")
    netrc_file = environment.USER_PATH / ".netrc"
    add_login = True
    if netrc_file.exists():
        with open(netrc_file, "r") as f:
            if "urs.earthdata.nasa.gov" in f.read():
                add_login = False

    if add_login:
        with open(environment.USER_PATH / ".netrc", "a+") as f:
            f.write(
                "\nmachine urs.earthdata.nasa.gov login {} password {}".format(
                    environment.EARTHDATA_USER, environment.EARTHDATA_PASSWORD
                )
            )
            os.fchmod(f.fileno(), 0o600)

    environment.EARTH_DATA_COOKIE_FILE.touch()
    subprocess.run(
        [
            "wget",
            "--load-cookies",
            environment.EARTH_DATA_COOKIE_FILE,
            "--save-cookies",
            environment.EARTH_DATA_COOKIE_FILE,
            "--keep-session-cookies",
            "https://urs.earthdata.nasa.gov",
        ],
        check=True,
    )


def _query_granule_metadata(bounds, product):
    granule_metadatas = [
        query(product=product, spatial=bound) for bound in bounds
    ]
    return pd.concat(granule_metadatas)


def _download_url(product, input):

    name, url = input
    outfile_path = environment.gedi_product_path(product) / name
    if os.path.exists(outfile_path):
        return outfile_path
    with tempfile.NamedTemporaryFile(
        dir=environment.gedi_product_path(product)
    ) as temp:
        subprocess.run(
            [
                "wget",
                "--load-cookies",
                environment.EARTH_DATA_COOKIE_FILE,
                "--save-cookies",
                environment.EARTH_DATA_COOKIE_FILE,
                "--auth-no-challenge=on",
                "--keep-session-cookies",
                "--content-disposition",
                "-O",
                temp.name,
                url,
            ],
            check=True,
        )
        shutil.move(temp.name, outfile_path)
    return outfile_path


def _parse_file(file_path):
    return gedi_database_loader.parse_file(file_path)


def _filter_file(geo_data_frame):
    return gedi_database_loader.filter_granules(geo_data_frame, None)


def _write_db(product, gedi_data):
    # Write all shots in the gedi dataframe in transaction while inserting the granule name into the granules table.
    # As long as the assumption that no two gedi granules hold the same shots, this ensures uniqueness of the shots
    # in the database without requiring a unique key constraint on the shot table. This is important since it allows
    # us to drop indexes and key constraints on the shots table during # the insert which increases performance
    # considerably.
    with _get_engine().begin() as con:
        if gedi_data.empty:
            return
        gedi_data = gedi_data.astype({"shot_number": "int64"})
        granule_name = gedi_data["granule_name"].head(1).item()
        granules_entry = pd.DataFrame(
            data={
                "granule_name": [granule_name],
                "created_date": [pd.Timestamp.utcnow()],
            }
        )
        # these steps are done for every granule
        granules_entry.to_sql(
            name=_granules_table(product),
            con=con,
            index=False,
            if_exists="append",
        )

        # Change strings to indices for efficient storage

        # beam_type: coverage == 0, full == 1
        beam_type_values = ['coverage', 'full']
        # Create a dictionary mapping stratum values to their index positions
        beam_type_to_index_map = {value: idx for idx, value in enumerate(beam_type_values)}
        # Replace the byte stratum with their corresponding indices using map
        gedi_data['beam_type'] = gedi_data['beam_type'].map(beam_type_to_index_map)

        # predict_stratum
        stratum_values = [b'DBT_Af', b'DBT_Au', b'DBT_Eu', b'DBT_NAs', b'DBT_SA', b'DBT_SAs', b'DBT_NAm',
                    b'EBT_Af', b'EBT_Au', b'EBT_Eu', b'EBT_NAs', b'EBT_SA', b'EBT_SAs', b'EBT_NAm',
                    b'ENT_Af', b'ENT_Au', b'ENT_Eu', b'ENT_NAs', b'ENT_SA', b'ENT_SAs', b'ENT_NAm',
                    b'DNT_Af', b'DNT_Au', b'DNT_Eu', b'DNT_NAs', b'DNT_SA', b'DNT_SAs', b'DNT_NAm',
                    b'GSW_Af', b'GSW_Au', b'GSW_Eu', b'GSW_NAs', b'GSW_SA', b'GSW_SAs', b'GSW_NAm']
        # Create a dictionary mapping stratum values to their index positions
        stratum_to_index_map = {value: idx for idx, value in enumerate(stratum_values)}
        # Replace the byte stratum with their corresponding indices using map
        gedi_data['predict_stratum'] = gedi_data['predict_stratum'].map(stratum_to_index_map)

        gedi_data = gedi_data.drop(columns=['granule_name', 'beam_name'])

        # making sure of proper types

        # general data
        gedi_data['shot_number'] = gedi_data['shot_number'].astype(np.int64)
        gedi_data['delta_time'] = gedi_data['delta_time'].astype(np.float64)
        # absolute_time is fine as it is
        gedi_data['predict_stratum'] = gedi_data['predict_stratum'].astype(np.uint8)
        # quality data
        gedi_data['beam_type'] = gedi_data['beam_type'].astype(np.uint8)
        gedi_data['sensitivity'] = gedi_data['sensitivity'].astype(np.float32)
        gedi_data['degrade_flag'] = gedi_data['degrade_flag'].astype(np.uint8)
        gedi_data['predictor_limit_flag'] = gedi_data['predictor_limit_flag'].astype(np.uint8)
        gedi_data['response_limit_flag'] = gedi_data['response_limit_flag'].astype(np.uint8)
        gedi_data['l4_quality_flag'] = gedi_data['l4_quality_flag'].astype(np.uint8)
        # Geo data
        gedi_data['lat_lowestmode'] = gedi_data['lat_lowestmode'].astype(np.float64)
        gedi_data['lon_lowestmode'] = gedi_data['lon_lowestmode'].astype(np.float64)
        gedi_data['elev_lowestmode'] = gedi_data['elev_lowestmode'].astype(np.float32)
        # biomass data
        gedi_data['agbd_t'] = gedi_data['agbd_t'].astype(np.float32)
        gedi_data['agbd_t_se'] = gedi_data['agbd_t_se'].astype(np.float32)
        # GEDI independent data
        gedi_data['landsat_treecover'] = gedi_data['landsat_treecover'].astype(np.float32)
        gedi_data['urban_focal_window_size'] = gedi_data['urban_focal_window_size'].astype(np.uint8)
        gedi_data['urban_proportion'] = gedi_data['urban_proportion'].astype(np.float32)
        # geometry is fine as it is
        
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(gedi_data.drop(columns=['absolute_time', 'geometry']).dtypes)
        #     print(gedi_data.drop(columns=['delta_time']).dtypes)
        #     # print(gedi_data[columns_parquet].iloc[0, :])

        # terrible practice, I'm sorry haha
        parquet_path = '/Users/remotelogin2/data/data_intermediate/{}.parquet'.format(granule_name)
        gedi_data.drop(columns=['absolute_time', 'geometry']).to_parquet(parquet_path)

        gedi_data.drop(columns=['delta_time']).to_postgis(
            name=_product_table(product),
            con=con,
            index=False,
            if_exists="append",
        )
        del gedi_data
    return granules_entry


def _product_table(product):
    if product == constants.GediProduct.L4A:
        return "level_4a"
    if product == constants.GediProduct.L2B:
        return "level_2b"
    raise ValueError("No product table defined for the product " + product)


def _granules_table(product):
    return _product_table(product) + "_granules"


def _query_downloaded(table_name):
    return pd.read_sql_table(
        table_name=table_name, columns=["granule_name"], con=_get_engine()
    )


def exec_spark(
    bounds: List[gpd.GeoSeries],
    product: constants.GediProduct,
    download_only: bool,
    dry_run: bool,
):
    if not os.path.exists(environment.EARTH_DATA_COOKIE_FILE):
        _fetch_cookies()

    granule_metadata = _query_granule_metadata(bounds, product).drop_duplicates(
        subset="granule_name"
    )
    print("Total granules found: ", len(granule_metadata.index) - 1)
    print("Total file size (MB): ", granule_metadata["granule_size"].sum())

    if dry_run:
        with tempfile.NamedTemporaryFile(
            dir="/tmp", mode="w+t", delete=False
        ) as temp:
            print(
                "Dry run: saving the names of all found granules to {}.\n".format(
                    temp.name
                )
            )
            for name in list(granule_metadata["granule_name"]):
                temp.write(name + "\n")
            temp.flush()
        if download_only:
            return

    if download_only:
        required_granules = granule_metadata
    else:
        stored_granules = _query_downloaded(_granules_table(product))
        required_granules = granule_metadata.loc[
            ~granule_metadata["granule_name"].isin(
                stored_granules["granule_name"]
            )
        ]

    if required_granules.empty:
        print("All granules for this region already present in the database")
        return

    print("Granules to download: ", len(required_granules.index) - 1)
    print(
        "File size to download (MB): ",
        required_granules["granule_size"].sum(),
    )

    if dry_run:
        return
    # if download_only:
    #     input("To proceed to download this data, press ENTER >>> ")
    # else:
    #     input("To proceed to download AND INGEST this data, press ENTER >>> ")
    if not os.path.exists(environment.gedi_product_path(product)):
        print(
            "Creating directory {}".format(environment.gedi_product_path(product))
        )
        os.mkdir(environment.gedi_product_path(product))

    # Spark starts here
    spark = (
        SparkSession.builder.config("spark.executor.memory", "10g")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    name_url = required_granules[["granule_name", "granule_url"]].to_records(
        index=False
    )
    urls = spark.sparkContext.parallelize(name_url)
    # Download the files to a stable location and return the file name
    files = urls.map(partial(_download_url, product))

    if download_only:
        files.count()
    else:
        # Parse each file into a geo data frame
        parsed_files = files.map(_parse_file)
        # Filter the geodataframe for suitable shots (l4_quality_flag == 1)
        filtered_files = parsed_files.map(_filter_file)
        # coalesce to 8 partitions to avoid overloading the database with many connections.
        # The number 8 was chosen sort of arbitrarily, could increase or decrease.
        out = filtered_files.coalesce(8).map(partial(_write_db, product))
        # count forces evaluation of the rdd
        out.count()

    spark.stop()
    print("done")

    # reinsert database index on points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and ingest GEDI data"
    )
    parser.add_argument(
        "--shapefile",
        help="Shapefile (zip) containing the world region to download.",
        type=str,
    )
    parser.add_argument(
        "--product",
        help="Name of GEDI product. Currently supports 'L4a' and 'L2b'.",
        type=str,
    )
    parser.add_argument(
        "--dry_run",
        help=("Dry run only: save all found granules to temporary file."),
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--download_only",
        help=(
            "Only download the raw granule files to shared location."
            "Do not also ingest the data into PostGIS database."
        ),
        action=argparse.BooleanOptionalAction,
    )
    parser.set_defaults(download_only=False)
    args = parser.parse_args()

    shapefile = pathlib.Path(args.shapefile)
    if not shapefile.exists():
        print("Unable to locate file {}".format(shapefile))
        exit(1)
    shp = gpd.read_file(shapefile)
    try:
        try:
            shp = check_and_format_shape(shp)
        except DetailError as exc:
            input(
                (
                    "The NASA API can only accept up to 5000 vertices in a single shape,\n"
                    "but the shape you supplied has {} vertices.\n"
                    "If you would like to automatically simplify this shape to its\n"
                    "bounding box, press ENTER, otherwise Ctrl-C to quit."
                ).format(exc.n_coords)
            )
            shp = check_and_format_shape(shp, simplify=True)
    except ValueError:
        print("This script only accepts one (multi)polygon at a time.")
        print("Please split up each row of your shapefile into its own file.")
        exit(1)

    product_str = args.product
    if not product_str:
        print("Must supply a GEDI product string.")
        print("e.g. --product=L4a")
        exit(1)
    if product_str.lower() == "l4a":
        product = constants.GediProduct.L4A
    elif product_str.lower() == "l2b":
        product = constants.GediProduct.L2B
    else:
        print("Product {} not supported".format(product_str))
        print("Please use one of L4A or L2B")
        exit(1)

    exec_spark(
        [shp],
        product,
        download_only=args.download_only,
        dry_run=args.dry_run,
    )
