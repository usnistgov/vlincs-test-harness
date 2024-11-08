from setuptools import setup

setup(
    name='vlincs-test-harness',
    version='1.0',
    packages=['leaderboards'],
    url='https://github.com/usnistgov/vlincs-test-harness',
    license='',
    author='Tim Blattner',
    author_email='timothy.blattner@nist.gov',
    install_requires=[
    'wheel',
    'google-api-python-client',
    'google-auth-httplib2',
    'google-auth-oauthlib',
    'jsonpickle',
    'jsonschema',
    'spython',
    'hypothesis-jsonschema',
    'pid',
    'numpy',
    'pytablewriter',
    'dominate',
    'GitPython',
    'httplib2',
    'scikit-learn',
    'airium',
    'pandas',
    'matplotlib',
    'msgpack'
    ],
    description='VLINCS Test Harness'
)
