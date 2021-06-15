import yaml


def p0_printer(par):
    iproc = par.rank

    def printer(*args, **kwargs):
        if iproc == 0:
            print(*args, **kwargs)

    return printer


def parse_ic(fname):
    """Parse the Nalu yaml input file for the initial conditions"""
    with open(fname, "r") as stream:
        try:
            dat = yaml.full_load(stream)
            u0 = float(
                dat["realms"][0]["initial_conditions"][0]["value"]["velocity"][0]
            )
            rho0 = float(
                dat["realms"][0]["material_properties"]["specifications"][0]["value"]
            )
            mu = float(
                dat["realms"][0]["material_properties"]["specifications"][1]["value"]
            )
            turb_model = dat["realms"][0]["solution_options"]["turbulence_model"]

            return u0, rho0, mu, turb_model

        except yaml.YAMLError as exc:
            print(exc)


# From: https://github.com/pandas-dev/pandas/issues/38425
def groupby_isclose(series, atol=0, rtol=0):
    # Sort values to make sure values are monotonically increasing:
    s = series.sort_values()

    # Calculate tolerance value:
    tolerance = atol + rtol * s

    # Calculate a monotonically increasing index that increase when the
    # differnce between current and previous value changes:
    by = s.diff().fillna(0).gt(tolerance).cumsum()
    # s_old = s.shift().fillna(s)
    # by = ((s - s_old).abs() > tolerance).cumsum().sort_index()

    return by
