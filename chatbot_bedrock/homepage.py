import streamlit as st

from components.navigation import page_navigation
from components.sidebar import sidebar
import backend as demo

page_navigation()
select_event, max_token_limit, total_usage_price = sidebar()