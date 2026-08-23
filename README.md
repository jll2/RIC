# Tools for calculating decay of radiation-induced ionic conductivity in quartz and similar materials

## Motivation

Materials that are naturally insulators can temporarily become
conductors after a dose of radiation.  Immediately after irradiation,
this radiation-induced-conductivity (RIC) is caused by free electrons and holes.  After a short
time, often mere nanoseconds, the free electrons and holes become
trapped, and the remaining conductivity is often due to free ions
diffusing through the material.  This conductivity is measurable for
minutes or even hours after irradiation ends.  In quartz, the ions are
alkalis and their diffusion occurs largely a single crystal axis.  The
python modules herein provide implementations of some special functions
needed to calculate the decay of RIC with time.

## Python scripts

There are two scripts here:

    ric_meijer.py
    ric_adaptive.py

Both scripts the provide the same functions but use different
implementations.  ric_meijer.py calculates the special functions using
the Meijer G functions from the mpmath package.  ric_adaptive.py
calculates using adaptive integration.  The Meijer G approach may be
considered more elegant but, at present, the adaptive integration
approach appears to be faster.  This may change depending on the
versions and/or hardware that you are using.

## References

The special functions provided by these scripts are used in the
following papers:

    "Decay of radiation–induced–conductivity analyzed as a problem in
    one–dimensional ion diffusion," by John L Lawless, Radiation Measurements
    (2026):107733.  https://doi.org/10.1016/j.radmeas.2026.107733

    "An explanation for why measurements of alkali ion mobility in quartz have
    been so inconsistent," by John L Lawless, in preparation (2026)

