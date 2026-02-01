#!/usr/bin/env python3
"""
City to Data USA Geo ID Mapper
Maps city names to Data USA MSA (Metropolitan Statistical Area) geo IDs for local market data queries.
MSA codes provide better data coverage than individual cities.
"""

from typing import Optional
import re

# Data USA Geo ID format: "31000US{MSA_CODE}"
# MSA codes from Census Bureau Metropolitan Statistical Areas
CITY_GEO_MAP = {
    # Major Tech Hubs
    "los angeles": "31000US31080",      # Los Angeles-Long Beach-Anaheim, CA MSA
    "la": "31000US31080",
    "los angeles ca": "31000US31080",
    "los angeles california": "31000US31080",
    
    "new york": "31000US35620",          # New York-Newark-Jersey City, NY-NJ-PA MSA
    "nyc": "31000US35620",
    "new york city": "31000US35620",
    "new york ny": "31000US35620",
    "manhattan": "31000US35620",
    "brooklyn": "31000US35620",
    
    "san francisco": "31000US41860",     # San Francisco-Oakland-Berkeley, CA MSA
    "sf": "31000US41860",
    "san francisco ca": "31000US41860",
    "oakland": "31000US41860",
    "san jose": "31000US41940",          # San Jose-Sunnyvale-Santa Clara, CA MSA
    "silicon valley": "31000US41940",
    
    "seattle": "31000US42660",           # Seattle-Tacoma-Bellevue, WA MSA
    "seattle wa": "31000US42660",
    
    "austin": "31000US12420",            # Austin-Round Rock-Georgetown, TX MSA
    "austin tx": "31000US12420",
    
    "denver": "31000US19740",            # Denver-Aurora-Lakewood, CO MSA
    "denver co": "31000US19740",
    
    "boston": "31000US14460",            # Boston-Cambridge-Newton, MA-NH MSA
    "boston ma": "31000US14460",
    "cambridge": "31000US14460",
    
    "chicago": "31000US16980",           # Chicago-Naperville-Elgin, IL-IN-WI MSA
    "chicago il": "31000US16980",
    
    "atlanta": "31000US12060",           # Atlanta-Sandy Springs-Alpharetta, GA MSA
    "atlanta ga": "31000US12060",
    
    "dallas": "31000US19100",           # Dallas-Fort Worth-Arlington, TX MSA
    "dallas tx": "31000US19100",
    "fort worth": "31000US19100",
    
    "phoenix": "31000US38060",          # Phoenix-Mesa-Chandler, AZ MSA
    "phoenix az": "31000US38060",
    
    "miami": "31000US33100",            # Miami-Fort Lauderdale-Pompano Beach, FL MSA
    "miami fl": "31000US33100",
    
    "washington": "31000US47900",       # Washington-Arlington-Alexandria, DC-VA-MD-WV MSA
    "dc": "31000US47900",
    "washington dc": "31000US47900",
    "arlington": "31000US47900",
    
    "philadelphia": "31000US37980",     # Philadelphia-Camden-Wilmington, PA-NJ-DE-MD MSA
    "philadelphia pa": "31000US37980",
    
    "houston": "31000US26420",          # Houston-The Woodlands-Sugar Land, TX MSA
    "houston tx": "31000US26420",
    
    "detroit": "31000US19820",          # Detroit-Warren-Dearborn, MI MSA
    "detroit mi": "31000US19820",
    
    "minneapolis": "31000US33460",       # Minneapolis-St. Paul-Bloomington, MN-WI MSA
    "minneapolis mn": "31000US33460",
    "twin cities": "31000US33460",
    
    "portland": "31000US38900",         # Portland-Vancouver-Hillsboro, OR-WA MSA
    "portland or": "31000US38900",
    
    "san diego": "31000US41740",        # San Diego-Chula Vista-Carlsbad, CA MSA
    "san diego ca": "31000US41740",
    
    "tampa": "31000US45300",            # Tampa-St. Petersburg-Clearwater, FL MSA
    "tampa fl": "31000US45300",
    
    "nashville": "31000US34980",        # Nashville-Davidson--Murfreesboro--Franklin, TN MSA
    "nashville tn": "31000US34980",
    
    "charlotte": "31000US16740",        # Charlotte-Concord-Gastonia, NC-SC MSA
    "charlotte nc": "31000US16740",
    
    "orlando": "31000US36740",          # Orlando-Kissimmee-Sanford, FL MSA
    "orlando fl": "31000US36740",
    
    "raleigh": "31000US39580",          # Raleigh-Cary, NC MSA
    "raleigh nc": "31000US39580",
    
    "indianapolis": "31000US26900",     # Indianapolis-Carmel-Anderson, IN MSA
    "indianapolis in": "31000US26900",
    
    "columbus": "31000US18140",         # Columbus, OH MSA
    "columbus oh": "31000US18140",
    
    "sacramento": "31000US40900",       # Sacramento-Roseville-Folsom, CA MSA
    "sacramento ca": "31000US40900",
    
    "kansas city": "31000US28140",      # Kansas City, MO-KS MSA
    "kansas city mo": "31000US28140",
    
    "milwaukee": "31000US33340",        # Milwaukee-Waukesha, WI MSA
    "milwaukee wi": "31000US33340",
    
    "cincinnati": "31000US17140",       # Cincinnati, OH-KY-IN MSA
    "cincinnati oh": "31000US17140",
    
    "pittsburgh": "31000US38300",       # Pittsburgh, PA MSA
    "pittsburgh pa": "31000US38300",
    
    "cleveland": "31000US17460",        # Cleveland-Elyria, OH MSA
    "cleveland oh": "31000US17460",
    
    "las vegas": "31000US29820",        # Las Vegas-Henderson-Paradise, NV MSA
    "las vegas nv": "31000US29820",
    "vegas": "31000US29820",
    
    "salt lake city": "31000US41620",   # Salt Lake City, UT MSA
    "salt lake city ut": "31000US41620",
    "slc": "31000US41620",
    
    "baltimore": "31000US12580",        # Baltimore-Columbia-Towson, MD MSA
    "baltimore md": "31000US12580",
    
    "richmond": "31000US40060",         # Richmond, VA MSA
    "richmond va": "31000US40060",
    
    "buffalo": "31000US15380",          # Buffalo-Cheektowaga, NY MSA
    "buffalo ny": "31000US15380",
    
    "providence": "31000US39300",       # Providence-Warwick, RI-MA MSA
    "providence ri": "31000US39300",
    
    "jacksonville": "31000US27260",     # Jacksonville, FL MSA
    "jacksonville fl": "31000US27260",
    
    "memphis": "31000US32820",          # Memphis, TN-MS-AR MSA
    "memphis tn": "31000US32820",
    
    "louisville": "31000US31140",       # Louisville/Jefferson County, KY-IN MSA
    "louisville ky": "31000US31140",
    
    "oklahoma city": "31000US36420",    # Oklahoma City, OK MSA
    "oklahoma city ok": "31000US36420",
    
    "hartford": "31000US25540",         # Hartford-East Hartford-Middletown, CT MSA
    "hartford ct": "31000US25540",
    
    "tucson": "31000US46060",           # Tucson, AZ MSA
    "tucson az": "31000US46060",
    
    "fresno": "31000US23420",           # Fresno, CA MSA
    "fresno ca": "31000US23420",
    
    "tulsa": "31000US46140",            # Tulsa, OK MSA
    "tulsa ok": "31000US46140",
    
    "honolulu": "31000US26180",         # Urban Honolulu, HI MSA
    "honolulu hi": "31000US26180",
    
    "omaha": "31000US36540",            # Omaha-Council Bluffs, NE-IA MSA
    "omaha ne": "31000US36540",
    
    "albany": "31000US10580",           # Albany-Schenectady-Troy, NY MSA
    "albany ny": "31000US10580",
    
    "birmingham": "31000US13820",       # Birmingham-Hoover, AL MSA
    "birmingham al": "31000US13820",
    
    "rochester": "31000US40380",        # Rochester, NY MSA
    "rochester ny": "31000US40380",
    
    "grand rapids": "31000US24340",     # Grand Rapids-Kentwood, MI MSA
    "grand rapids mi": "31000US24340",
    
    "albany": "31000US10580",           # Albany-Schenectady-Troy, NY MSA
    "albany ny": "31000US10580",
    
    "tallahassee": "31000US45220",      # Tallahassee, FL MSA
    "tallahassee fl": "31000US45220",
    
    "boise": "31000US14260",            # Boise City, ID MSA
    "boise id": "31000US14260",
    
    "spokane": "31000US44060",          # Spokane-Spokane Valley, WA MSA
    "spokane wa": "31000US44060",
    
    "des moines": "31000US19780",       # Des Moines-West Des Moines, IA MSA
    "des moines ia": "31000US19780",
    
    "wichita": "31000US48620",          # Wichita, KS MSA
    "wichita ks": "31000US48620",
    
    "madison": "31000US31540",          # Madison, WI MSA
    "madison wi": "31000US31540",
    
    "little rock": "31000US30780",      # Little Rock-North Little Rock-Conway, AR MSA
    "little rock ar": "31000US30780",
    
    "anchorage": "31000US11260",        # Anchorage, AK MSA
    "anchorage ak": "31000US11260",
    
    "reno": "31000US39900",             # Reno, NV MSA
    "reno nv": "31000US39900",
    
    "baton rouge": "31000US12940",      # Baton Rouge, LA MSA
    "baton rouge la": "31000US12940",
    
    "knoxville": "31000US28940",        # Knoxville, TN MSA
    "knoxville tn": "31000US28940",
    
    "greensboro": "31000US24660",       # Greensboro-High Point, NC MSA
    "greensboro nc": "31000US24660",
    
    "charleston": "31000US16700",       # Charleston-North Charleston, SC MSA
    "charleston sc": "31000US16700",
    
    "colorado springs": "31000US17820", # Colorado Springs, CO MSA
    "colorado springs co": "31000US17820",
    
    "lexington": "31000US30460",        # Lexington-Fayette, KY MSA
    "lexington ky": "31000US30460",
    
    "syracuse": "31000US45060",         # Syracuse, NY MSA
    "syracuse ny": "31000US45060",
    
    "dayton": "31000US19380",           # Dayton-Kettering, OH MSA
    "dayton oh": "31000US19380",
    
    "akron": "31000US10420",            # Akron, OH MSA
    "akron oh": "31000US10420",
    
    "tulsa": "31000US46140",            # Tulsa, OK MSA
    "tulsa ok": "31000US46140",
    
    "new orleans": "31000US35300",      # New Orleans-Metairie, LA MSA
    "new orleans la": "31000US35300",
    
    "bridgeport": "31000US14860",       # Bridgeport-Stamford-Norwalk, CT MSA
    "bridgeport ct": "31000US14860",
    
    "worcester": "31000US49340",       # Worcester, MA-CT MSA
    "worcester ma": "31000US49340",
    
    "allentown": "31000US10900",       # Allentown-Bethlehem-Easton, PA-NJ MSA
    "allentown pa": "31000US10900",
    
    "albany": "31000US10580",          # Albany-Schenectady-Troy, NY MSA
    "albany ny": "31000US10580",
    
    "harrisburg": "31000US25420",      # Harrisburg-Carlisle, PA MSA
    "harrisburg pa": "31000US25420",
    
    "youngstown": "31000US49660",      # Youngstown-Warren-Boardman, OH-PA MSA
    "youngstown oh": "31000US49660",
    
    "scranton": "31000US42540",        # Scranton--Wilkes-Barre, PA MSA
    "scranton pa": "31000US42540",
    
    "trenton": "31000US45940",         # Trenton-Princeton, NJ MSA
    "trenton nj": "31000US45940",
    
    "fort wayne": "31000US23060",      # Fort Wayne, IN MSA
    "fort wayne in": "31000US23060",
    
    "savannah": "31000US42340",        # Savannah, GA MSA
    "savannah ga": "31000US42340",
    
    "mobile": "31000US33660",          # Mobile, AL MSA
    "mobile al": "31000US33660",
    
    "shreveport": "31000US43340",      # Shreveport-Bossier City, LA MSA
    "shreveport la": "31000US43340",
    
    "augusta": "31000US12260",         # Augusta-Richmond County, GA-SC MSA
    "augusta ga": "31000US12260",
    
    "chattanooga": "31000US16860",     # Chattanooga, TN-GA MSA
    "chattanooga tn": "31000US16860",
    
    "spartanburg": "31000US43900",     # Spartanburg, SC MSA
    "spartanburg sc": "31000US43900",
    
    "flint": "31000US22420",           # Flint, MI MSA
    "flint mi": "31000US22420",
    
    "santa barbara": "31000US42060",   # Santa Maria-Santa Barbara, CA MSA
    "santa barbara ca": "31000US42060",
    
    "bakersfield": "31000US12540",     # Bakersfield, CA MSA
    "bakersfield ca": "31000US12540",
    
    "stockton": "31000US44700",       # Stockton, CA MSA
    "stockton ca": "31000US44700",
    
    "modesto": "31000US33700",        # Modesto, CA MSA
    "modesto ca": "31000US33700",
    
    "visalia": "31000US47300",        # Visalia, CA MSA
    "visalia ca": "31000US47300",
    
    "salinas": "31000US41500",        # Salinas, CA MSA
    "salinas ca": "31000US41500",
    
    "santa rosa": "31000US42220",     # Santa Rosa-Petaluma, CA MSA
    "santa rosa ca": "31000US42220",
    
    "vallejo": "31000US46700",        # Vallejo-Fairfield, CA MSA
    "vallejo ca": "31000US46700",
    
    "merced": "31000US32900",         # Merced, CA MSA
    "merced ca": "31000US32900",
    
    "yuba city": "31000US49700",      # Yuba City, CA MSA
    "yuba city ca": "31000US49700",
    
    "hanford": "31000US25260",        # Hanford-Corcoran, CA MSA
    "hanford ca": "31000US25260",
    
    "el centro": "31000US20940",      # El Centro, CA MSA
    "el centro ca": "31000US20940",
    
    "redding": "31000US39820",        # Redding, CA MSA
    "redding ca": "31000US39820",
    
    "chico": "31000US17020",          # Chico, CA MSA
    "chico ca": "31000US17020",
    
    "eureka": "31000US21700",         # Eureka-Arcata-Fortuna, CA MSA
    "eureka ca": "31000US21700",
    
    "san luis obispo": "31000US42020", # San Luis Obispo-Paso Robles, CA MSA
    "san luis obispo ca": "31000US42020",
    
    "santa cruz": "31000US42100",     # Santa Cruz-Watsonville, CA MSA
    "santa cruz ca": "31000US42100",
    
    "napa": "31000US34900",           # Napa, CA MSA
    "napa ca": "31000US34900",
    
    "yuma": "31000US49740",           # Yuma, AZ MSA
    "yuma az": "31000US49740",
    
    "prescott": "31000US39140",       # Prescott Valley-Prescott, AZ MSA
    "prescott az": "31000US39140",
    
    "flagstaff": "31000US22380",      # Flagstaff, AZ MSA
    "flagstaff az": "31000US22380",
    
    "lake havasu": "31000US29420",    # Lake Havasu City-Kingman, AZ MSA
    "lake havasu az": "31000US29420",
    
    "sierra vista": "31000US43420",   # Sierra Vista-Douglas, AZ MSA
    "sierra vista az": "31000US43420",
    
    "show low": "31000US43300",       # Show Low, AZ MSA
    "show low az": "31000US43300",
    
    "nogales": "31000US35740",        # Nogales, AZ MSA
    "nogales az": "31000US35740",
    
    "payson": "31000US37740",         # Payson, AZ MSA
    "payson az": "31000US37740",
    
    "safford": "31000US40940",        # Safford, AZ MSA
    "safford az": "31000US40940",
    
    "kingman": "31000US29420",        # Lake Havasu City-Kingman, AZ MSA
    "kingman az": "31000US29420",
    
    "douglas": "31000US43420",        # Sierra Vista-Douglas, AZ MSA
    "douglas az": "31000US43420",
    
    "phoenix mesa": "31000US38060",   # Phoenix-Mesa-Chandler, AZ MSA
    "phoenix mesa az": "31000US38060",
    
    "tucson": "31000US46060",         # Tucson, AZ MSA
    "tucson az": "31000US46060",
    
    "yuma": "31000US49740",           # Yuma, AZ MSA
    "yuma az": "31000US49740",
    
    "prescott valley": "31000US39140", # Prescott Valley-Prescott, AZ MSA
    "prescott valley az": "31000US39140",
    
    "flagstaff": "31000US22380",      # Flagstaff, AZ MSA
    "flagstaff az": "31000US22380",
    
    "lake havasu city": "31000US29420", # Lake Havasu City-Kingman, AZ MSA
    "lake havasu city az": "31000US29420",
    
    "sierra vista": "31000US43420",   # Sierra Vista-Douglas, AZ MSA
    "sierra vista az": "31000US43420",
    
    "show low": "31000US43300",       # Show Low, AZ MSA
    "show low az": "31000US43300",
    
    "nogales": "31000US35740",        # Nogales, AZ MSA
    "nogales az": "31000US35740",
    
    "payson": "31000US37740",         # Payson, AZ MSA
    "payson az": "31000US37740",
    
    "safford": "31000US40940",        # Safford, AZ MSA
    "safford az": "31000US40940",
    
    "kingman": "31000US29420",        # Lake Havasu City-Kingman, AZ MSA
    "kingman az": "31000US29420",
    
    "douglas": "31000US43420",        # Sierra Vista-Douglas, AZ MSA
    "douglas az": "31000US43420",
}

# Deduplicate entries while preserving first occurrence
def _dedupe_city_geo_map(city_map: dict) -> dict:
    deduped = {}
    for key, value in city_map.items():
        if key not in deduped:
            deduped[key] = value
    return deduped

CITY_GEO_MAP = _dedupe_city_geo_map(CITY_GEO_MAP)

def get_geo_id(city: str) -> Optional[str]:
    """
    Convert city name to Data USA geo ID (MSA code).
    
    Args:
        city: City name (e.g., "Los Angeles", "New York", "San Francisco, CA")
        
    Returns:
        Data USA geo ID string (e.g., "31000US31080") or None if not found
        
    Examples:
        >>> get_geo_id("Los Angeles")
        '31000US31080'
        >>> get_geo_id("NYC")
        '31000US35620'
        >>> get_geo_id("Unknown City")
        None
    """
    if not city:
        return None
    
    # Normalize: lowercase, strip whitespace, remove common suffixes
    normalized = city.lower().strip()
    
    # Remove common suffixes that might interfere
    normalized = re.sub(r'\s*,\s*(ca|california|ny|new york|tx|texas|fl|florida|wa|washington|co|colorado|ma|massachusetts|il|illinois|ga|georgia|az|arizona|pa|pennsylvania|mi|michigan|mn|minnesota|or|oregon|tn|tennessee|nc|north carolina|in|indiana|oh|ohio|nv|nevada|ut|utah|md|maryland|va|virginia|ri|rhode island|ct|connecticut|ok|oklahoma|hi|hawaii|ne|nebraska|al|alabama|ia|iowa|ks|kansas|wi|wisconsin|ar|arkansas|ak|alaska|la|louisiana|sc|south carolina|ky|kentucky|nj|new jersey|id|idaho|de|delaware|me|maine|nh|new hampshire|vt|vermont|mt|montana|wy|wyoming|nd|north dakota|sd|south dakota|wv|west virginia|ms|mississippi|mo|missouri)$', '', normalized)
    
    # Try direct lookup
    geo_id = CITY_GEO_MAP.get(normalized)
    if geo_id:
        return geo_id
    
    # Try partial matches (e.g., "san francisco bay area" -> "san francisco")
    for key, value in CITY_GEO_MAP.items():
        if key in normalized or normalized in key:
            return value
    
    return None

def normalize_city_name(city: str) -> str:
    """
    Normalize city name for consistent lookup.
    
    Args:
        city: Raw city input
        
    Returns:
        Normalized city name
    """
    if not city:
        return ""
    
    normalized = city.lower().strip()
    # Remove common prefixes/suffixes
    normalized = re.sub(r'^(city of|town of|village of)\s+', '', normalized)
    normalized = re.sub(r'\s+(city|town|village)$', '', normalized)
    
    return normalized
