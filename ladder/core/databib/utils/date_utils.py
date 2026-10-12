from datetime import date, datetime

from dateutil import parser


def parse_datetime(a_date: str) -> datetime:
    if type(a_date) is str:
        return consume_time_zone(parser.parse(a_date))
    raise ValueError('Invalid date: ' + a_date)


def parse_date(a_date: str) -> date:
    return parse_datetime(a_date).date()


def format_datetime(a_date: datetime) -> str:
    return a_date.strftime('%Y-%m-%dT%H:%M:%SZ')


def consume_time_zone(a_date: datetime):
    utc_offset = a_date.utcoffset()
    if utc_offset:
        # local_time = utc_time + utc_offset <=> utc_time = local_time - utc_offset
        return a_date.replace(tzinfo=None) - utc_offset

    if a_date.tzinfo is not None:
        return a_date.replace(tzinfo=None)

    return a_date
