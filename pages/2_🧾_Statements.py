# Need to add ratio analysis

from common import *
common_styles()
sidebar()

from io import BytesIO, StringIO
from zipfile import ZipFile

import numpy as np
import os
import streamlit as st
from streamlit_extras.no_default_selectbox import selectbox
import datetime
import pandas as pd
#from yahooquery import Ticker
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List

def format_num(x):

	# first try converting to string
	if type(x) == str:
		try:
			x = float(x)
		except:
			pass

	if type(x) in [int, float]:
		if np.isnan(x):
			x = None
		elif abs(x)>=1_000_000_000:
			x = f'{x/1_000_000_000:.1f}B'
		elif abs(x)>=1_000_000:
			x = f'{x/1_000_000:.1f}M'
		elif abs(x)>=1_000:
			x = f'{x/1_000:.1f}K'
		else:
			x = f'{x:.1f}'

	return x

def add_percentage(x):
	if not np.isnan(x):
		x = f"{x*100:.1f}%"
	
	return x

def remove_percentage_sign(x):
	return x.replace("%", "")

def prettify(option: str) -> str:
	words = option.split("_")
	words = [word.title() for word in words]

	return " ".join(words)

@st.cache_data
def zip_data(_ticker, symbol, online, *args):
	balance_sheet_df, cash_flow_df, income_statement_df = read_data(_ticker, symbol, online, *args)
	
	buf = BytesIO()
	with ZipFile(buf, "x") as csv_zip:
		csv_zip.writestr(f"{symbol}_Balance_Sheet.csv", pd.DataFrame(balance_sheet_df.T).to_csv())
		csv_zip.writestr(f"{symbol}_Cash_Flow.csv", pd.DataFrame(cash_flow_df.T).to_csv())
		csv_zip.writestr(f"{symbol}_Income_Statement.csv", pd.DataFrame(income_statement_df.T).to_csv())
	return buf.getvalue()

@st.cache_data
def convert_df_to_csv(df):
	return df.to_csv()

@st.cache_data
def convert_dfs_to_excel(dfs: list, sheet_names: list):
	buffer = BytesIO()
	with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
		for df, sheet_name in zip(dfs, sheet_names):
			df.T.to_excel(writer, sheet_name)
	return buffer

@st.cache_data
def convert_df_to_html(df):
	return df.to_html()

@st.cache_data
def convert_fig_for_download(fig, config):
	buffer = StringIO()
	fig.write_html(buffer, include_plotlyjs=True, config=config, full_html=False)
	return buffer.getvalue().encode()

@st.cache_data
def get_helper():
	return pd.read_csv("./helper.csv", index_col="Variable")["Meaning"]

@st.cache_data
def read_data(_ticker, symbol, online, *args):
	statements = ["balance_sheet", "cash_flow", "income_statement"]

	if online:
		if not os.path.exists("./data"):
			os.mkdir("./data")

		for statement in statements:
			try:
				# df = _ticker.all_financial_data()
				statement_df = getattr(_ticker, statement)(*args)
				# cash_flow_df = _ticker.cash_flow()
				# income_statement_df = _ticker.income_statement()
			except TypeError:
				st.error("Did not work!")
					
			col = statement_df.columns[0]
			if "date" in col.lower():
				statement_df = statement_df.rename(columns = {col: "date"}).reset_index().set_index("date")
			
			file = f"./data/{symbol}_{statement}.csv"
			statement_df.to_csv(file, index=True)
	
	try:
		balance_sheet_df = pd.read_csv(f"./data/{symbol}_balance_sheet.csv", index_col="date", parse_dates=True)
		cash_flow_df = pd.read_csv(f"./data/{symbol}_cash_flow.csv", index_col="date", parse_dates=True)
		income_statement_df = pd.read_csv(f"./data/{symbol}_income_statement.csv", index_col="date", parse_dates=True)
	except Exception:
		display_backup_missing()
		return None

	return balance_sheet_df, cash_flow_df, income_statement_df

@st.cache_data
def get_statements_data(_ticker, symbol, online, *args):
	balance_sheet_df, cash_flow_df, income_statement_df = read_data(_ticker, symbol, online, *args)

	balance_sheet_df = balance_sheet_df.drop(columns=["symbol", "periodType", "currencyCode"])
	
	balance_sheet_df = balance_sheet_df.rename(columns={
		"StockholdersEquity": "TotalEquity"
	})
	
	balance_sheet_df["TotalCurrentLiabilities"] = balance_sheet_df["CurrentLiabilities"] + balance_sheet_df["OtherCurrentLiabilities"]
	balance_sheet_df["CurrentEquity"] = balance_sheet_df["CurrentLiabilities"] + balance_sheet_df["OtherCurrentLiabilities"]
	
	balance_sheet_df.loc["Statement"] = "Balance Sheet"

	cash_flow_df = cash_flow_df.drop(columns=[
		"NetIncome", "symbol", "periodType", "currencyCode"
	])
	cash_flow_df.loc["Statement"] = "Cash Flow"

	income_statement_df = income_statement_df.drop(columns=["symbol", "periodType", "currencyCode"])

	income_statement_df = income_statement_df.rename(columns={
		"PretaxIncome": "EBT",
		"TaxRateForCalcs": "EffectiveTaxRate"
	})

	income_statement_df.loc["Statement"] = "Income Statement"
	
	df = (
		balance_sheet_df
		.merge(cash_flow_df, left_index=True, right_index=True)
		.merge(income_statement_df, left_index=True, right_index=True)
	)
	
	# df = df.fillna(0)

	df = df.T
	df.index.name = "Variable"
	
	return df

@st.cache_data
def ratio_analysis(ratio_df, ratios, percentage_ratios):
	ratio_df = ratio_df.copy().T.drop("Statement")
	
	ratio_df[ratios[0]] = (ratio_df["EBIT"]/ratio_df["TotalAssets"])
	ratio_df[ratios[1]] = (
		ratio_df["EBIT"]*(1-ratio_df["EffectiveTaxRate"])
		)/(
		ratio_df["TotalAssets"]
	)
	
	ratio_df[ratios[2]] = (ratio_df["NetIncome"]/ratio_df["TotalEquity"])
	# Here we are using total equity; otherwise, we can use "commonstockequity" -> Common equity = shareholder's equity (or total equity) – preference shares
	
	ratio_df[ratios[3]] = ratio_df["EBIT"]/ratio_df["CurrentEquity"]
	ratio_df[ratios[4]] = (
		ratio_df["EBIT"]*(1-ratio_df["EffectiveTaxRate"])
	)/ratio_df["CurrentEquity"]

	ratio_df[ratios[5]] = (ratio_df["EBITDA"]/ratio_df["OperatingRevenue"])
	ratio_df[ratios[6]] = (ratio_df["EBIT"]/ratio_df["OperatingRevenue"])
	ratio_df[ratios[7]] = (ratio_df["EBT"]/ratio_df["OperatingRevenue"])
	ratio_df[ratios[8]] = (ratio_df["NetIncome"]/ratio_df["OperatingRevenue"])
	ratio_df[ratios[9]] = (ratio_df["OperatingRevenue"]/ratio_df["TotalAssets"])
	ratio_df[ratios[10]] = ratio_df["TotalAssets"]/ratio_df["TotalEquity"]

	ratio_df[ratios[10]] = ratio_df["TotalAssets"]/ratio_df["TotalEquity"]
	ratio_df[ratios[11]] = ratio_df["NetIncome"]/ratio_df["EBT"]
	ratio_df[ratios[12]] = ratio_df["EBT"]/ratio_df["EBIT"]

	ratio_df[ratios[13]] = ratio_df["OperatingRevenue"]/ratio_df["CurrentAssets"]
	ratio_df[ratios[14]] = ratio_df["OperatingRevenue"]/ratio_df["TotalNonCurrentAssets"]
	ratio_df[ratios[15]] = ratio_df["OperatingRevenue"]/ratio_df["NetPPE"] # Since using the gross equipment values would be misleading, it's recommended to use the net asset value that's reported on the balance sheet by subtracting the accumulated depreciation from the gross.
	ratio_df[ratios[16]] = ratio_df["OperatingRevenue"]/ratio_df["TotalEquity"]
	
	ratio_df[ratios[17]] = ratio_df["OperatingRevenue"]/ratio_df["WorkingCapital"]
	ratio_df[ratios[18]] = ratio_df["OperatingRevenue"]/ratio_df["Inventory"]
	ratio_df[ratios[19]] = 365/ratio_df[ratios[18]]
	ratio_df[ratios[20]] = ratio_df["OperatingRevenue"]/ratio_df["AccountsReceivable"]
	ratio_df[ratios[21]] = 365/ratio_df[ratios[20]]
	cash_expenses_per_day = (ratio_df["TotalExpenses"]/365)
	ratio_df[ratios[22]] = ratio_df["CashAndCashEquivalents"]/cash_expenses_per_day
	ratio_df[ratios[23]] = pd.Series(dtype=object) # ratio_df["MaterialConsumed"]/ratio_df["AccountsPayable"] # (MaterialConsumed is a missing entity in yahoo finance)
	ratio_df[ratios[24]] = 365/ratio_df[ratios[23]]
	ratio_df[ratios[25]] = ratio_df[ratios[19]] + ratio_df[ratios[21]] + ratio_df[ratios[22]] - ratio_df[ratios[24]]

	ratio_df[ratios[26]] = ratio_df["TotalDebt"]/ratio_df["TotalEquity"]
	ratio_df[ratios[27]] = ratio_df["TotalDebt"]/(ratio_df["TotalDebt"] + ratio_df["TotalEquity"])
	ratio_df[ratios[28]] = ratio_df["TotalEquity"]/(ratio_df["TotalDebt"] + ratio_df["TotalEquity"])
	ratio_df[ratios[29]] = ratio_df["EBIT"]/ratio_df["InterestExpense"]
	ratio_df[ratios[30]] = ratio_df["EBIT"]/(ratio_df["InterestExpense"] + ratio_df["TotalDebt"])

	ratio_df[ratios[31]] = pd.Series(dtype=object) # ratio_df["DividendDeclared"]/ratio_df["ShareIssued"] # (DividendDeclared is a missing entity in yahoo finance)
	ratio_df[ratios[32]] = ratio_df["NetIncome"]/ratio_df["ShareIssued"]
	ratio_df[ratios[33]] = pd.Series(dtype=object) # ratio_df["CashDividendsPaid"]/ratio_df[ratios[39]]
	ratio_df[ratios[34]] = ratio_df[ratios[31]]/ratio_df[ratios[32]]
	ratio_df[ratios[35]] = 1 - ratio_df[ratios[34]]

	ratio_df[ratios[36]] = ratio_df["CurrentAssets"]/ratio_df["TotalCurrentLiabilities"]
	ratio_df[ratios[37]] = (ratio_df["CurrentAssets"] - ratio_df["Inventory"])/ratio_df["TotalCurrentLiabilities"]

	ratio_df[ratios[38]] = ratio_df["TotalEquity"]/ratio_df["ShareIssued"]
	ratio_df[ratios[39]] = pd.Series(dtype=object) # (CurrentMarketPrice to be taken from stock data)
	ratio_df[ratios[40]] = ratio_df[ratios[39]]/ratio_df[ratios[32]]
	ratio_df[ratios[41]] = ratio_df[ratios[39]]/ratio_df[ratios[38]]
	
	try:
		ratio_df[percentage_ratios] = ratio_df[percentage_ratios].applymap(add_percentage)
	except:
		pass

	return ratio_df

def horizontal_analysis(column, first_value):
	column = column/first_value
	return column

def vertical_analysis(row, total_assets):
	row = row/total_assets
	return row

analyses = {
	"": "Absolute",
	"Hor": "Horizontal",
	"Ratio": "Ratio",
	"Ver": "Vertical"
}

def format_analysis(analysis):
	return analyses[analysis]

def main(tickers, symbol, strings: dict, online: bool):
	st.header("Financial Statements")

	frequency = "Annual"
	arg = frequency[:1].lower()

	df = get_statements_data(tickers, symbol, online, arg)
	
	dfs_to_download = read_data(tickers, symbol, online, arg)
	sheet_names = ["Balance Sheet", "Cash Flow", "Income Statement"]

	c1, c2, c3 = st.columns([1, 1, 1])

	with c1:
		st.session_state["selected_analyses"] = st.multiselect(
			label = "Analyses",
			#label_visibility = "collapsed",
			options = sorted(analyses.keys()),
			format_func = format_analysis
		)

	with c2:
		st.session_state["selected_statements"] = st.multiselect(
				label = "Statements",
				options = sorted(df["Statement"].dropna().unique() if any( item in ["", "Hor", "Ver"] for item in st.session_state["selected_analyses"] ) else []),
				format_func = prettify
				# label_visibility = "collapsed",
				#index = statements.index("balance_sheet"),
		)
	
	ratios = [
		"Return on Assets (ROA/ROTA) (Before Tax)",
		"Return on Assets (ROA/ROTA) (After Tax)",
		"Return on Equity (ROE)",
		"Return on Current Equity (ROCE) (Before Tax)",
		"Return on Current Equity (After Tax)",
		
		"EBITDA Margin",
		"EBIT Margin/OPM",
		"EBT Margin",
		"Net Profit Margin (NPM)",
		"Asset Turnover Ratio (ATR)",

		"Total Leverage",
		"Tax Factor",
		"Interest Factor",

		"Current Asset Turnover Ratio",
		"Non-Current Asset Turnover Ratio",
		"PPE Utilisation Ratio/ Capital Intensity Ratio",
		"Equity Turnover Ratio",

		"Working Capital Turnover Ratio",
		"Inventory Turnover Ratio (ITR)",
		"Days Inventory",
		"Debtors Turnover Ratio (DTR)",
		"Days Debtors/Receivables/Average Collection Period",
		"Days Cash",
		"Creditor Turnover Ratio (CTR)",
		"Days Creditors/Payables/Average Payment Period",
		"Cash Conversion Cycle (Days)",

		"Debt/Equity Ratio",
		"Debt Ratio /Debt Capitalisation Ratio",
		"Equity Ratio/ Equity Capitalisation Ratio",
		"Interest Coverage Ratio",
		"Total Debt Service Ratio",

		"Dividend Per Share",
		"Earning Per Share",
		"Dividend Yeild Ratio",
		"Dividend Payout Ratio (D/P Ratio)",
		"Retension Ratio",

		"Current Ratio",
		"Quick Ratio/ Acid Test Ratio",

		"Book Value per Share",
		"Market Value Per Share (on balance sheet date)",
		"Price Earning Ratio (P/E)",
		"Price to Book Value Ratio (P/B)",
	]

	percentage_ratios = [
		ratios[index]
		for index in [
			0, 1, 2, 3, 4,
			5, 6, 7, 8,
			27, 28,
			33, 35,
		]
	]

	ratio_groups = {
		"Overall Performance Ratio": list(range(0, 4+1)),
		"Profit Margin Ratios": list(range(5, 9+1)),
		"Two Factor Dupont Analysis": [0],
		"Three Factor DuPont": [8, 9, 10, 2],
		"Five Factor Dupont": [11, 12, 6, 9, 10, 2],
		"Turnover/Efficiency Ratios": [9] + list(range(13, 16+1)),
		"Working Capital Ratio": list(range(17, 25+1)),
		"Insolvency Ratio": list(range(26, 30+1)),
		"Test of Dividend Policy": list(range(31, 35+1)),
		"Liquidity Ratios": [36, 37],
		"Valuation Ratios": [38, 39, 32, 40, 41],
	}

	with c3:
		st.session_state["selected_ratio_groups"] = st.multiselect(
				label = "Ratio Groups",
				options = sorted(ratio_groups.keys() if "Ratio" in st.session_state["selected_analyses"] else []),
				format_func = prettify
				# label_visibility = "collapsed",
				#index = statements.index("balance_sheet"),
		)
		
	# if len(st.session_state["selected_statements"]) == 0:
	# 	st.session_state["selected_statements"] = statements
	# else:
	copy = df.copy()
	ratio_df = copy.pipe(ratio_analysis, ratios, percentage_ratios)
	
	df = df[
		df["Statement"].isin(
			[prettify(selected_statement) for selected_statement in st.session_state["selected_statements"]]
		)
	]

	df = df.T.drop(["Statement"]).astype(float)
	filtered_df = pd.DataFrame(index = df.index)

	if "" in st.session_state["selected_analyses"]:
		filtered_df = filtered_df.merge(
			df, # .add_suffix("Absolute")
			how="inner", left_index=True, right_index=True
		)
	
	zero_cols = []
	null_cols = []
	if "Hor" in st.session_state["selected_analyses"]:
		
		# fixing divide by 0
		col = df.iloc[0]
		
		zero_cols = df.columns[col==0]
		non_zero_cols = df.columns[col!=0]

		null_cols = df.columns[np.isnan(col)]
	
		# df = df[non_zero_cols]
		
		horizontal_df = df[non_zero_cols]
		col = horizontal_df.iloc[0]

		filtered_df = filtered_df.merge(
			horizontal_df.apply(horizontal_analysis, axis=1, args=[col]).applymap(add_percentage).add_suffix('Hor'),
			how="inner", left_index=True, right_index=True
		)
	
	ratio_attributes = []
	
	if "Ratio" in st.session_state["selected_analyses"]:		
		ratio_attributes = []

		for group in st.session_state["selected_ratio_groups"]:
			for attr in ratio_groups[group]:
				ratio_attributes.append(ratios[attr])
		
		ratio_df = ratio_df[ratio_attributes]
			
		filtered_df = filtered_df.merge(
			ratio_df,
			how="inner", left_index=True, right_index=True
		)

	if "Ver" in st.session_state["selected_analyses"]:
		copy = df.T.merge(copy[["Statement"]], left_index=True, right_index=True)

		if "Balance Sheet" in st.session_state["selected_statements"]:
			filtered_df = filtered_df.merge(
				(
					copy[
						copy["Statement"] == "Balance Sheet"
					]
					.drop(columns="Statement")
					.T
					.apply(vertical_analysis, axis=0, args=[df["TotalAssets"]])
					.applymap(add_percentage)
					.add_suffix('Ver')
				),
				how="inner", left_index=True, right_index=True
			)
		if "Income Statement" in st.session_state["selected_statements"]:
			filtered_df = filtered_df.merge(
				(
					copy[
						copy["Statement"] == "Income Statement"
					]
					.drop(columns="Statement")
					.T
					.apply(vertical_analysis, axis=0, args=[df["TotalRevenue"]])
					.applymap(add_percentage)
					.add_suffix('Ver')
				),
				how="inner", left_index=True, right_index=True
			)

		if "Cash Flow" in st.session_state["selected_statements"]:
			st.warning("Cashflow statement does not have vertical analysis", icon="⚠️")
			return
		
	regular_attributes = list(df.columns.values)
	
	attributes_options = []			
	if len(st.session_state["selected_analyses"]) == 0:
		pass
	else:
		if any(
			item in st.session_state["selected_statements"] for item in ["Balance Sheet", "Cash Flow", "Income Statement"]
		):
			attributes_options += regular_attributes
		if "Ratio" in st.session_state["selected_analyses"]:
			attributes_options += ratio_attributes

	c1, c2, c3 = st.columns([2, 2, 1])

	with c1:
		st.session_state["selected_attributes"] = st.multiselect(
			label = "Attributes",
			#label_visibility = "collapsed",
			options = sorted(attributes_options)
		)

	checkpoints = {
		"2020-01-29": "Covid UK First Case",
		"2020-11-26": "Covid UK Lockdown Ends"
	}
	with c2:
		st.session_state["selected_checkpoints"] = st.multiselect(
			label = "Checkpoints",
			options = checkpoints.keys(),
			format_func = lambda x: checkpoints[x]
		)
	
	with c3:
		st.write("")
		st.write("")
		st.session_state["simplify_graph"] = st.checkbox(
			"Simplify Graph"
		)	

	for attr in st.session_state["selected_attributes"]:
		for col in zero_cols:
			if attr == col:
				st.warning(f"First value of {col} is 0")
				return
		for col in null_cols:
			if attr == col:
				st.warning(f"First value of {col} is missing.")
				return

	fig = None

	if len(st.session_state["selected_analyses"]) == 0:
		pass
	elif len(st.session_state["selected_attributes"]) == 0:
		pass
	else:
		# attributes_to_expand = [
		# 	attr if attr not in ratios for attr in st.session_state["selected_attributes"]
		# ]
		# attributes_to_expand
		# return
		# .remove(None)

		# analyses_to_expand = [
		# 	ana if ana not in ratios else None
		# 	for ana in st.session_state["selected_analyses"]
		# ].remove(None)
		
		selected_ratios = [
			att
			for att in st.session_state["selected_attributes"]
			if att in ratios
		]

		attributes_to_modify = [
			att
			for att in st.session_state["selected_attributes"]
			if att not in ratios
		]

		analyses_to_modify = [
			att
			for att in st.session_state["selected_analyses"]
			if att != "Ratio"
		]

		modified_attributes = [
			att + ana
			for att in attributes_to_modify
			for ana in analyses_to_modify
		]

		st.session_state["selected_attributes_modified"] = modified_attributes + selected_ratios

		# st.session_state["selected_attributes_modified"]
		# return

		filtered_df = filtered_df[st.session_state["selected_attributes_modified"]]

		plot_data = filtered_df.copy().astype(str)
		table_data = filtered_df.copy()
		
		if len(st.session_state["selected_attributes_modified"]) > 0:
			fig = px.line(
				plot_data,
				y = st.session_state["selected_attributes_modified"],
				markers = True
			)

			for checkpoint in st.session_state["selected_checkpoints"]:
				fig.add_vline(
					x=checkpoint,
					line_color = "grey",
					line_dash="dot"
				)

			fig.update_layout(
				margin=dict(t=80, r=0, b=0, l=0),
				
				# axes titles
				xaxis_title = None,
				yaxis_title = None,
				
				hovermode = "x unified",
				
				# legend
				# showlegend = False,
				legend = dict(
					title = "",
					groupclick="toggleitem",
					orientation = 'h',
					
					# positioning
					x = 0,
					xanchor = "left",
					
					y = 1.1,
					yanchor = "bottom",
					
					font = dict(
						size = 10
					),
					itemsizing = 'constant',
					
					# click behavior
					#itemclick = 'toggleothers',
					#itemdoubleclick = 'toggle'
				)
			)

			if st.session_state["simplify_graph"] == True:
				annotations = []

				for i, d in enumerate(fig.data):
					trace_values = []
					for y in d.y:
						if y == "nan" or y is None:
							trace_values.append(None)
						else:
							trace_values.append(y)

					first_index = pd.Series(trace_values).first_valid_index()
					last_index = pd.Series(trace_values).last_valid_index()
					
					first_value = d.y[first_index]
					last_value = d.y[last_index]

					first_text = '  ' + d.name + ' ' + format_num(first_value) + '  '
					last_text = '  ' + format_num(last_value) + ' ' + f"({format_num(float(remove_percentage_sign(last_value)) - float(remove_percentage_sign(first_value)))})"  + '  '

					annotations.append(dict(
						x = d.x[first_index],
						y = remove_percentage_sign(first_value),
						xanchor='right',
						yanchor='middle',
						text=first_text, # + ' {}%'.format(d.y[0]),
						font = dict(
							color = d.line.color,
						),
						showarrow = False,
						align = "right",
						# bgcolor = "hsla(0, 100%, 100%, 1)"
					))

					annotations.append(dict(
						x = d.x[last_index],
						y = remove_percentage_sign(last_value),
						xanchor='left',
						yanchor='middle',
						text = last_text, # + ' {}%'.format(d.y[0]),
						font = dict(
							color = d.line.color,
						),
						showarrow = False,
						align = "left"
						# bgcolor = "hsla(0, 100%, 100%, 1)"
					))
				
				hide_all_axis_stuff = dict(
						showgrid=False,
						zeroline=True,
						showline=False,
						showticklabels=False,
				)

				fig.update_layout(
					margin=dict(t=0, r=0, b=0, l=0),

					annotations=annotations,
					showlegend=False,

					xaxis = hide_all_axis_stuff,
					yaxis = hide_all_axis_stuff,

					height = 600
				)

		if len(st.session_state["selected_attributes_modified"]) > 0:
			st.plotly_chart(
				fig,
				use_container_width=True,
				config = config,
			)

		table_data.index = pd.to_datetime(table_data.index).strftime("%Y")
		table_data = (
			table_data
			.T
			.sort_index()
			.applymap(format_num)
		)

		with st.expander("Table View"):
			st.dataframe(
				table_data,
				use_container_width=True
			)

		dfs_to_download = list(dfs_to_download)
		dfs_to_download.append(table_data)

		sheet_names.append("Filtered_Data")

		if len(st.session_state["selected_attributes_modified"]) > 0:
			definitions = get_helper()
			definitions = definitions[
				definitions.index.isin(st.session_state["selected_attributes"])
			]

			with st.expander("Legend"):
				st.dataframe(
					definitions,
					use_container_width=True
				)

	with st.sidebar:
		st.divider()
		if st.button('Prepare Downloads'):
			st.download_button(
				label="Save Data",
				data = convert_dfs_to_excel(
					dfs_to_download,
					sheet_names
				),
				file_name=f"{symbol}.xlsx"
			)
	
			if fig:
				st.download_button(
					label='Save Graph',
					data=convert_fig_for_download(fig, config),
					file_name=f'{symbol}.html',
				)

main(st.session_state["tickers"], st.session_state["symbol"], st.session_state["strings"], st.session_state["online"])
