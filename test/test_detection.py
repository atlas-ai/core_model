#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from detection import event_detection
import pandas as pd
import read_data_n_param as rdp


chunks ={
    "LTT": {
        "start": "",
        "end": ""
    },
    "RTT": {

    },
}

turn_threshold = 0.8
lane_change_threshold = 0.6
samp_rate = 50
evt_param = rdp.read_evt_param("detection_coefficients.csv")


def helper_data(start, end):
    """Returns test data in a give time frame (start, end)"""
    rot_z = pd.Series.from_csv('./test/full_track_20180108_50Hz/rot_z.csv')
    lat = pd.Series.from_csv('./test/full_track_20180108_50Hz/lat.csv')
    long = pd.Series.from_csv('./test/full_track_20180108_50Hz/long.csv')
    alt = pd.Series.from_csv('./test/full_track_20180108_50Hz/alt.csv')
    crs = pd.Series.from_csv('./test/full_track_20180108_50Hz/crs.csv')
    spd = pd.Series.from_csv('./test/full_track_20180108_50Hz/spd.csv')

    rot_z = select_chunk(rot_z, start, end)
    lat = select_chunk(lat, start, end)
    long = select_chunk(long, start, end)
    alt = select_chunk(alt, start, end)
    crs = select_chunk(crs, start, end)
    spd = select_chunk(spd, start, end)

    return rot_z, lat, long, alt, crs, spd


def select_chunk(dataframe, start, end):
    """start and end time format "2018-01-08 09:51:29.713000"
    """
    return dataframe[(dataframe.index > start) &
                     (dataframe.index < end)]


def test_LTT():
    rot_z, lat, long, alt, crs, spd = helper_data(chunks["LTT"]["start"], chunks["LTT"]["end"])
    df_evt = event_detection(rot_z, lat, long, alt, crs, spd, evt_param, samp_rate,
                             turn_threshold, lane_change_threshold)
    assert df_evt['type'].tolist() == []


def test_RTT():
    rot_z, lat, long, alt, crs, spd = helper_data(chunks["RTT"]["start"], chunks["RTT"]["end"])
    df_evt = event_detection(rot_z, lat, long, alt, crs, spd, evt_param, samp_rate,
                             turn_threshold, lane_change_threshold)
    assert df_evt['type'].tolist() == []


def test_event_summary():
    assert 1 == 1