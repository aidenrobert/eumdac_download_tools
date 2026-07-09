#!/usr/bin/env python
# coding: utf-8

"""
Written by: Aiden Jönsson, 2026

Download tools for the EUMDAC data store.
Helper functions for accessing & processing data.
"""

def distance_to_site(lat, lon, site_lat, site_lon):

    """
    Calculates great-circle distance between points and a specified site

    Inputs:
    lat, lon : array-like
        Latitude and longitude of data points
    site_lat, site_lon : float
        Target location

    Returns:
    distance_km : ndarray
        Distance in km
    """

    import numpy as np
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    _, _, distance_m = geod.inv(
        np.full_like(lon, site_lon),
        np.full_like(lat, site_lat),
        lon,
        lat
    )

    return distance_m / 1000

def subset_radius(ds, site_lat, site_lon, radius_km):

    """
    Subsets data within a given distance from a specified site

    Inputs:
    ds : xarray dataset
        Data to subset with latitude, longitude coordinates
    site_lat, site_lon : float
        Target location
    radius_km : float
        Radius to retain data from

    Returns:
    ds_sub : xarray dataset
        Dataset with added variable for distance to site
    """

    ## Calculate distance to site and mask out values outside of given radius
    dist = distance_to_site(
        ds.latitude.values,
        ds.longitude.values,
        site_lat,
        site_lon
    )
    mask = dist <= radius_km
    ds_sub = ds.isel(groups=mask)

    ## Store distance as a variable
    ds_sub = ds_sub.assign(distance_to_site_km=("groups", dist[mask]))

    return ds_sub