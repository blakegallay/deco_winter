Things that have to do with calculating and analyzing Hillas parameters. 

Some good resources:
  http://ihp-lx.ethz.ch/Stamet/magic/parameters.html
  https://www.mpi-hd.mpg.de/hfm/HESS/pages/publications/proceedings/Conf_Palaiseau_2005/deNaurois.pdf
  
My slides: https://docs.google.com/presentation/d/1bN6viRP7E_aP9MXYbzJE9Bm6t2qdwNnSHxLOU3rKU18/edit#slide=id.p

Directories and files:
/irradiation_images: images from the irradiation experiment, used to test Hillas scripts

/results: all sorts of results, calculations, and plots

hillas_deco.py, hillas_irradiation.py, hillas_simulation.py: used to calculate and save Hillas parameters (along with CNN classification) for events

Spot_imagepaths.csv, Track_imagepaths.csv: csvs containing paths on the Cobalts to all DECO Spots and Tracks

zoom_predictions_4class.py/pyc: original CNN module
zoom_predictions_4class_modified.py/pyc: CNN module changed to read image arrays instead of jpg files

