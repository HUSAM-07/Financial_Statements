from common import *
common_styles()
sidebar()
import os
import streamlit as st
import datetime
import pandas as pd
from yahooquery import Ticker
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List

def get_stock_data(tickers, history_args, symbol, online: str):
	
	args_list = [arg for arg in history_args.values() if arg is not None]

	file = f"./data/{symbol}_{'_'.join(args_list)}.csv"

	index_col = "date"

	if online:
		df = st.session_state["tickers"].history(**history_args)

		if index_col in df.index.names:
			df = df.reset_index().set_index(index_col)
		
		if not os.path.exists("./data"):
			os.mkdir("./data")
		
		df.to_csv(file, index=True)
	else:
		try:
			df = pd.read_csv(file, index_col=index_col, parse_dates=True)
		except Exception:
			display_backup_missing()
			return None
	return df

def stocks(tickers, symbol, strings: dict, online):
	"""Provides an illustration of the `Ticker.history` method

	Arguments:
		tickers {Ticker} -- A yahaooquery Ticker object
		symbol {List[str]} -- A list of symbol
	"""
	st.header("Historical Pricing")
	
	history_args = {
		"period": "1y",
		"interval": "1d",
		"start": datetime.datetime.now() - datetime.timedelta(days=365),
		"end": None,
	}

	c1, c2, c3 = st.columns([1, 1, 1])

	with c1:
		option_1 = st.selectbox("Period or Start / End Dates", ["Period", "Dates"], 0)
	
	if option_1 == "Period":
		with c2:
			history_args["period"] = st.selectbox(
				"Period", options=Ticker.PERIODS, index=7  # pylint: disable=protected-access
			)

		history_args["start"] = None
		history_args["end"] = None
	else:
		with c2:
			history_args["start"] = st.date_input("Start Date", value=history_args["start"])
			history_args["end"] = st.date_input("End Date")
		
		history_args["period"] = None

	with c3:
		history_args["interval"] = st.selectbox(
				"Interval", options=Ticker.INTERVALS, index=8  # pylint: disable=protected-access
			)
	
	df = get_stock_data(tickers, history_args, symbol, online)
	
	fig = go.Figure(go.Ohlc(
		x=df.index,
		open=df['open'],
		high=df['high'],
		low=df['low'],
		close=df['close']
	))

	fig.update_xaxes(
		rangebreaks=[
			dict(bounds=["sat", "mon"]) #hide weekends
		]
	)

	st.plotly_chart(
		fig,
		use_container_width=True,
		config = config
	)

stocks(st.session_state["tickers"], st.session_state["symbol"], st.session_state["strings"], st.session_state["online"])