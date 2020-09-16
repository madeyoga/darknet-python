from setuptools import setup, find_packages
setup(
    name="darknet_yolov3",
    version="16.09.2020",
    packages=find_packages(),

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
        "numpy",
        "opencv-python",
        "matplotlib"],

    package_data={
        
    },

    # metadata to display on PyPI
    author="madeyoga",
    author_email="yeogaa02@gmail.com",
    description="A simple python script to load & use darknet yolov3 model using cv2.",
    keywords="darknet yolov3 python",
    url="https://github.com/madeyoga/darknet-python",
)
