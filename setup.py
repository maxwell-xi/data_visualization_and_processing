import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='data_visualization_and_processing',
    version='0.0.1',
    author='Maxwell XI',
    author_email='xi@itis.swiss',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/maxwell-xi/data_visualization_and_processing',
    project_urls = {
        "Bug Tracker": "https://github.com/maxwell-xi/data_visualization_and_processing/issues"
    },
    license='MIT',
    packages=['data_visualization_and_processing'],
    install_requires=['requests'],
)
