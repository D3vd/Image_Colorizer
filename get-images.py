import requests
import urllib.request

api_key = '4072f60bf9e8a68eb2ac5dfc08cd9bc4'
page_counter = 1
photo_url = "https://farm{}.staticflickr.com/{}/{}_{}.jpg"


def download_img(img_url, img_id):
        file_name = 'dataset/' + img_id + '.jpg'
        print('Downloading ' + img_id + '.jpg....')
        urllib.request.urlretrieve(img_url, file_name)


while True:
        response = requests.get('https://api.flickr.com/services/rest/', params={
                'method': 'flickr.photos.search',
                'format': 'json',
                'nojsoncallback': 1,
                'api_key': api_key,
                'tags': 'landscape',
                'tag_mode': 'all',
                'page': page_counter,
            }).json()

        photo_url = "https://farm{}.staticflickr.com/{}/{}_{}.jpg"

        for photo in response['photos']['photo']:
                url = photo_url.format(photo['farm'], photo['server'], photo['id'], photo['secret'])
                download_img(url, photo['id'])

        page_counter += 1
