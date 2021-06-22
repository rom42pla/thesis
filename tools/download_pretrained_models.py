import requests
import sys
import os
from os.path import join, exists


def download(url, filepath):
    with open(filepath, 'wb') as f:
        print(f"Downloading {filepath} from {url}")
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50 - done)))
                sys.stdout.flush()
    sys.stdout.write('\n')


if __name__ == "__main__":
    models_urls = {
        "pretrained_1shot.pth": "http://dl.yf.io/fs-det/models/voc/split1/tfa_cos_1shot/model_final.pth",
        "pretrained_2shot.pth": "http://dl.yf.io/fs-det/models/voc/split1/tfa_cos_2shot/model_final.pth",
        "pretrained_3shot.pth": "http://dl.yf.io/fs-det/models/voc/split1/tfa_cos_3shot/model_final.pth",
        "pretrained_5shot.pth": "http://dl.yf.io/fs-det/models/voc/split1/tfa_cos_5shot/model_final.pth",
        "pretrained_10shot.pth": "http://dl.yf.io/fs-det/models/voc/split1/tfa_cos_10shot/model_final.pth",
    }

    for model_name, model_url in models_urls.items():
        model_filepath = join("checkpoints", model_name)
        if exists(model_filepath):
            print(f"{model_filepath} already downloaded")
            continue
        try:
            download(url=model_url, filepath=model_filepath)
        except:
            if exists(model_filepath):
                os.remove(model_filepath)
                continue

    print(f"Pretrained models downloaded")
