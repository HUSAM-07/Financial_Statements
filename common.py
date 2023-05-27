import streamlit as st

def common_styles():
	st.set_page_config(layout="wide")
	common_styles = """
		<style>
		/*#MainMenu,*/
		footer
		{visibility: hidden; !important}
		</style>
		"""

	st.markdown(common_styles, unsafe_allow_html=True)

def display_backup_missing():
	st.error("Local backup not available; check for latest online version", icon="⚠️")

config = dict(
	doubleClickDelay = 400, # (ms) affects the single click delay; default = 300ms
	displayModeBar = False,
	displaylogo = False,
	showTips = False
)

def sidebar():
	if "symbol" not in st.session_state:
		st.session_state["symbol"] = "MANU"
		
	st.session_state["symbol"] = st.sidebar.text_input(
		"Ticker Symbol",
		value = "MANU"
	)

	if "online" not in st.session_state:
		st.session_state["online"] = False

	st.session_state["online"] = st.sidebar.checkbox("Latest?")

	st.session_state["asynchronous"] = True
	asynchronous_str = "" if not st.session_state["asynchronous"] else ", asynchronous=True"

	st.session_state["formatted"] = False
	formatted_str = "" if not st.session_state["formatted"] else ", formatted=True"

	st.session_state["tickers"] = init_ticker(
		st.session_state["symbol"],
		formatted=st.session_state["formatted"],
		asynchronous=st.session_state["asynchronous"]
	)

	st.session_state["strings"] = {
		'formatted_str': formatted_str,
		'asynchronous_str': asynchronous_str
	}


from yahooquery import Ticker

#@st.cache_data
def init_ticker(symbol, **kwargs):
	return Ticker(symbol, **kwargs)
