from setuptools import setup, find_packages

setup(
    name             = 'yolov4',
    version          = '1.0',
    description      = 'Yolov4 Module',
    author           = 'Chanwoo Gwon',
    author_email     = 'arknell@yonsei.ac.kr',
    url              = 'https://github.com/KChanwoo/ABR-image-processor.git',
    install_requires = [ ],
    packages         = find_packages(exclude = ['docs', 'tests*']),
    keywords         = ['ai', 'yolov4'],
    python_requires  = '>=3',
    zip_safe=False,
    classifiers      = [
        'Programming Language :: Python :: 3.6'
    ]
)