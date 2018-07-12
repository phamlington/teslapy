## Add something here

## Changelog
### Latest (July 12, 2018)
* fixed bugs in ales244 demo program
* added `dealias` filter to spectralLES, changed `RK4_integrate()` to use the `dealias` filter instead of `les_filter`, and added `les_filter`ing to the SGS stress functions

### March 12, 2018
* added np.seterr(divide='ignore') contexts around known divide-by-zero operations to suppress runtime warnings
* added argparse parser to spectralLES class
* added LoadFromFile(argparse.Action) class
* changed spectralLES constructor method arguments
* updated many spectralLES class method names
* updated computeSource keyword arguments
* changed `computeSource_linear_forcing()` to be able to just compute and return
  dvScale, and to accept dvScale from keyword arguments. This change improves
  runtime performance
