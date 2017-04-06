#import requests
#test = requests.get('http://musicbrainz.org/ws/2/artist/5b11f4ce-a62d-471e-81fc-a69a8278c7da?inc=aliases&fmt=json')
#test.json()

import musicbrainzngs

musicbrainzngs.set_useragent("class music analysis", "0.1" )
test = musicbrainzngs.get_artist_by_id('4ac4e32b-bd18-402e-adad-ae00e72f8d85')