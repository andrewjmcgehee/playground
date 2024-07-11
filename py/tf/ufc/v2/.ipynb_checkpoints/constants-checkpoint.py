ALL_EVENTS_URL = "http://ufcstats.com/statistics/events/completed?page=all"
EVENT_KWARGS = {
    "tag": "a",
    "field": "href",
    "get_field": True,
    "regex": "http.*event-details.*",
}
FIGHT_DATA_IGNORE = {3, 6, 8, 10, 11, 12}
FIGHT_DATA_SIZES = [
    None, 2, 4, None, 4, 4, None, 2, None, 4, None, None, None, 4, 4, 4, 4, 4, 4
]
FIGHT_KWARGS = {
    "tag": "tr",
    "field": "data-link",
    "get_field": True,
    "regex": "http.*fight-details.*"
}
FIGHT_RECORD_KEYS = [
    "fight_id", "datetime", "label", "is_draw", "red_id", "red_kd", "red_ss",
    "red_ss_attempted", "red_total", "red_total_attempted", "red_td",
    "red_td_attempted", "red_sub_attempted", "red_control_min",
    "red_control_sec", "red_ss_head", "red_ss_head_attempted", "red_ss_body",
    "red_ss_body_attempted", "red_ss_leg", "red_ss_leg_attempted",
    "red_ss_distance", "red_ss_distance_attempted", "red_ss_clinch",
    "red_ss_clinch_attempted", "red_ss_ground", "red_ss_ground_attempted",
    "blue_id", "blue_kd", "blue_ss", "blue_ss_attempted", "blue_total",
    "blue_total_attempted", "blue_td", "blue_td_attempted",
    "blue_sub_attempted", "blue_control_min", "blue_control_sec",
    "blue_ss_head", "blue_ss_head_attempted", "blue_ss_body",
    "blue_ss_body_attempted", "blue_ss_leg", "blue_ss_leg_attempted",
    "blue_ss_distance", "blue_ss_distance_attempted", "blue_ss_clinch",
    "blue_ss_clinch_attempted", "blue_ss_ground", "blue_ss_ground_attempted"
]
FIGHT_UNFORMATTED_COLUMNS = [
    "{}_kd", "{}_ss", "{}_ss_attempted", "{}_total", "{}_total_attempted",
    "{}_td", "{}_td_attempted", "{}_sub_attempted", "{}_control_min",
    "{}_control_sec", "{}_ss_head", "{}_ss_head_attempted", "{}_ss_body",
    "{}_ss_body_attempted", "{}_ss_leg", "{}_ss_leg_attempted",
    "{}_ss_distance", "{}_ss_distance_attempted", "{}_ss_clinch",
    "{}_ss_clinch_attempted", "{}_ss_ground", "{}_ss_ground_attempted"
]
FIGHTER_KWARGS = {
    "tag": "a",
    "field": "href",
    "get_field": True,
    "regex": "http.*fighter-details.*"
}
MAX_THREADS = 10
OMIT = {
    # fights
    "http://ufcstats.com/fight-details/8e03db41687d9132",
    "http://ufcstats.com/fight-details/b80e6a799c95d499",
    "http://ufcstats.com/fight-details/e4fe950846b51bdf",
    "http://ufcstats.com/fight-details/7ffcc3a72e082ace",
    "http://ufcstats.com/fight-details/6449a1a9a69a830c",
    "http://ufcstats.com/fight-details/635fbf57001897c7",
    "http://ufcstats.com/fight-details/a1db4c917777aa79",
    "http://ufcstats.com/fight-details/4b334c9727eee450",
    "http://ufcstats.com/fight-details/b297c3e938e1005e",
    "http://ufcstats.com/fight-details/f59b1215176636f6",
    "http://ufcstats.com/fight-details/c413b0abc04358c3",
    "http://ufcstats.com/fight-details/77bf1e37929b0d59",
    "http://ufcstats.com/fight-details/8b258bbb37f74a66",
    "http://ufcstats.com/fight-details/2f449bd58b3d9a99",
    "http://ufcstats.com/fight-details/565ecefd8a37ad7e",
    "http://ufcstats.com/fight-details/5701dbbbfa4f8313",
    "http://ufcstats.com/fight-details/4bce0ce561a65288",
    "http://ufcstats.com/fight-details/d93c8c77e1091a16",
    "http://ufcstats.com/fight-details/a5c90086fb65f58e",
    "http://ufcstats.com/fight-details/b80872821bc4f6ba",
    "http://ufcstats.com/fight-details/3badedeb2c5533f4"
}
ONE_DAY = 86400  # 1 day in seconds
RAW_EVENTS_BASE = "raw_html/events"
RAW_FIGHTERS_BASE = "raw_html/fighters"
RAW_FIGHTS_BASE = "raw_html/fights"