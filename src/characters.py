abnormal_hyphen_characters = "\u2043\u2010\u2011\u2013\u2014\u2212\u2012—"
abnormal_period_characters = "\uFF0E\u2024"
dot_characters = "\uFE52\u2027\u2E31\u00B7\u2022••◦◯"

ordered_list_incrementor = "([0-9Ⅰ-ⅿ]+|[a-z])"
ordered_list_separator = f"[\\)\\.\\-{abnormal_period_characters}{abnormal_hyphen_characters}{dot_characters}]"
ordered_list_prefix = f"^({ordered_list_incrementor}{ordered_list_separator})+$"

unordered_list_prefix = (
    f"^[\\-{abnormal_hyphen_characters}{dot_characters}▪◾■▫◆◇➤➲o]$"
)
