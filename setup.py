from setuptools import setup
setup(
    name="pyqsm",
    version="0.1",
    install_requires=[
        'numpy',
        'scipy',
        'pyunwrap3d @ git+https://@github.com/AlanKuurstra/pyunwrap3d.git#egg=pyunwrap3d',
    ],
    #this would have added the frequencyEstimate module into site-packages so that import would be
    #import frequencyEstimate instead of import pyqsm.frequencyEstimate
    #py_modules=['frequencyEstimate']

    #all the scripts are in the main directory because this repo is not really mean to be distributed
    #it's just testing out a couple QSM algorithms. So I just map the root directory to a package named pyqsm
    package_dir={'pyqsm':''},
    #now that I've created a package directory out of my root directory, I can add that package to be copied into
    #the site-packages
    #packages=['pyQSM'],
    #or instead of adding all the scripts, I can just add the one script that I want to distribute
    #note that since I've created a package dir, I can now import pyqsm.frequencyEstimate instead of just frequencyEstimate
    py_modules=['pyqsm.frequencyEstimate'],
)