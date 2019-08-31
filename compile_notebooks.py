"""
(본 스크립트는 튜토리얼과 관계 없는 개발용 코드 입니다.)
Ipynb 파일을 Python파일로 변환합니다.
아래와 같이 nbconvert를 써도 가능하지만, 불필요한 내용(# In[4]:) 들이 포함되어 있습니다.
```
for i in *.ipynb **/*.ipynb; do
    echo "$i"
    jupyter nbconvert --to script "$i" "$i"
done
```
이 스크립트는 불필요한 내용을 골라 삭제하고
각 폴더 내에 있는 모든 .ipynb파일을 .py파일로 변환합니다.
"""

import io
import os
import glob
import re
from nbconvert.exporters.script import ScriptExporter


def main():
    exporter = ScriptExporter()
    cell_pattern = "#\s(In)\[(.*)\]:\n\n\n"  #  # In [*]:
    comment_pattern = "(#(\s*)\n)"
    matplotlib_pattern = "(.*get_ipython.*)"

    for nbpath in glob.iglob('./[0-9]**/*.ipynb', recursive=True):
        base, ext = os.path.splitext(nbpath)
        script, resources = exporter.from_filename(nbpath)

        # remove unecessary patterns
        script = re.sub(cell_pattern, "", script)
        script = re.sub(comment_pattern, "", script)
        script = re.sub(matplotlib_pattern, "", script)
        pypath = (base + resources.get('output_extension', '.txt'))

        if os.path.exists(pypath):
            os.remove(pypath)
            print("Cleaned existing script %s" % pypath)
        with io.open(pypath, 'w', encoding='utf-8') as f:
            f.write(script)
        print("Saved script %s" % pypath)


if __name__ == "__main__":
    main()
