import json
from multiprocessing import Pool
import tqdm

reviewBooksPath = "C:/Users/yinkai/Desktop/tmp.json"
reviewBooksSavePath = "C:/Users/yinkai/Desktop/process.txt"

# reviewBooksPath = "./reviews_Books_5.json"
# reviewBooksSavePath = "./process.txt"

def processJSON(jsonstr):
    js = json.loads(jsonstr.strip())
    overall = js["overall"]
    reviewText = js["reviewText"]
    if overall == 5.0:
        return reviewText + "\t" + "1" + "\n"
    elif overall == 1.0:
        return reviewText + "\t" + "0" + "\n"
    else:
        return None

def process(path, newpath):
    """
    path: json文件路径
    newpath: 处理完保存路径
    """
    pool = Pool()
    with open(path, "r") as f:
        lines = f.readlines()
    
    results = []
    
    for r in tqdm.tqdm(pool.imap_unordered(processJSON, lines), total=len(lines)):
        if r is not None:
            results.append(r)

    with open(newpath, "w") as nf:
        nf.writelines(results)



if __name__ == "__main__":
    process(reviewBooksPath, reviewBooksSavePath)
