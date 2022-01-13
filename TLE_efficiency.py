#! python3

# Heat capacity butane
def cp_gas(feed_type, temperature):
    """
    temperature in degrees C back-regressed from PROII
    """
    if feed_type == "Butane":
        return (
            -9.0 * 10 ** (-4) * temperature ** 2
            + 2.7489 * temperature
            + 1761.8
        )
    elif feed_type == "Naphtha":
        return (
            -9.0 * 10 ** (-4) * temperature ** 2
            + 2.6691 * temperature
            + 1710.9
        )
    elif feed_type == "Ethane":
        return (
            -8.763 * 10 ** (-4) * temperature ** 2
            + 2.7116 * temperature
            + 2050.0
        )


def cp_gass(feed_types, temperatures):
    """
    heat capacity of multiple datapoints with a given feed type and
    temperature of the fluid
    """
    import pandas as pd

    if isinstance(temperatures, float) or isinstance(temperatures, int):
        return cp_gas(feed_types, temperatures)
    else:
        return pd.Series(
            [
                cp_gas(feed_type, temperature)
                for feed_type, temperature in zip(feed_types, temperatures)
            ],
            index=temperatures.index,
        )


def gass_visc(feed_types, temperatures):
    """
    Gas viscosity of multiple datapoints with a given feed type and temperature
    of the fluid
    """
    import pandas as pd

    if isinstance(temperatures, float) or isinstance(temperatures, int):
        return gas_visc(feed_types, temperatures)
    else:
        return pd.Series(
            [
                gas_visc(feed_type, temperature)
                for feed_type, temperature in zip(feed_types, temperatures)
            ],
            index=temperatures.index,
        )


def gass_thermal_cond(feed_types, temperatures):
    """
    Gas thermal conductivity of multiple datapoints with a given feed type and
    temperature of the fluid
    """
    import pandas as pd

    if isinstance(temperatures, float) or isinstance(temperatures, int):
        return gas_thermal_cond(feed_types, temperatures)
    else:
        return pd.Series(
            [
                gas_thermal_cond(feed_type, temperature)
                for feed_type, temperature in zip(feed_types, temperatures)
            ],
            index=temperatures.index,
        )


def heat_transferred(feed_type, flow, t_in, t_out):
    """
    heat transferred from flow in MW
    temps in degrees C
    """
    import numpy as np
    import pandas as pd

    cp_in = cp_gass(feed_type, t_in)
    cp_out = cp_gass(feed_type, t_out)
    #     print(cp_in)
    #     print(cp_out)
    # Apparently taking the average Cp works better (input from Nenad)
    # I disagree
    #     cp_in = cp_gass(
    #       feed_type,
    #       pd.concat((t_in, t_out), axis=1).mean(axis=1)
    #     )
    #     cp_out = cp_in
    return flow / 3.6 * (cp_in * t_in - cp_out * t_out) / 10 ** 6


def gas_thermal_cond(feed_type, temperature):
    if feed_type == "Butane":
        return (2 * 10 ** (-7) * temperature + 7 * 10 ** (-6)) * 10 ** 3
    elif feed_type == "Naphtha":
        return (2 * 10 ** (-7) * temperature + 6 * 10 ** (-6)) * 10 ** 3
    elif feed_type == "Ethane":
        return (
            2.447 * 10 ** (-8) * temperature ** 2
            + 1.8693 * 10 ** (-4) * temperature
            + 3.3693 * 10 ** (-2)
        )


def gas_visc(feed_type, temperature):
    """
    Temperature in degrees C
    """
    if feed_type == "Butane":
        return 3 * 10 ** (-8) * temperature + 1 * 10 ** (-5)
    elif feed_type == "Naphtha":
        return 3 * 10 ** (-8) * temperature + 0.87 * 10 ** (-5)
    # Incorrect still!
    elif feed_type == "Ethane":
        return 3 * 10 ** (-8) * temperature + 1 * 10 ** (-5)


def TLX_total_flow(flows_df):
    """Take the sum of all input flows for TLX"""
    return flows_df.sum(axis=1)


def alpha_i(
    feed_types,
    temperatures,
    flow,
    number_of_tubes_TLX,
    inner_diameter,
    heat_transfer_coeff_adj_spyro,
):
    """Fluid heat transfer coefficient"""
    Nu = Nusselt(
        feed_types, temperatures, flow, number_of_tubes_TLX, inner_diameter
    )
    lambda_ = (
        gass_thermal_cond(feed_types, temperatures)
        * heat_transfer_coeff_adj_spyro
    )

    return Nu * lambda_ / inner_diameter


def log_mean(x, y):
    import numpy as np

    return (y - x) / (np.log(y) - np.log(x))


def heat_cond_w(t_in, t_out, t_wall):
    return 24.4 + 0.0041 * (
        average_wall_temperature(t_in, t_out, t_wall) - 1250.7 + 2 * 273
    )


def average_wall_temperature(t_in, t_out, t_wall):
    return (log_mean(t_in, t_out) + t_wall) / 2


def U_clean_reciproce(
    feed_types,
    bulk_temperatures,
    t_in,
    t_out,
    t_wall,
    flow,
    number_of_tubes_TLX,
    inner_diameter,
    outer_diameter,
    heat_transfer_coeff_adj_spyro,
):
    import numpy as np

    thickness_w = outer_diameter - inner_diameter
    alpha_i_ = alpha_i(
        feed_types,
        bulk_temperatures,
        flow,
        number_of_tubes_TLX,
        inner_diameter,
        heat_transfer_coeff_adj_spyro,
    )
    lambda_w = heat_cond_w(t_in, t_out, t_wall)

    return 1 / alpha_i_ + thickness_w / lambda_w * inner_diameter / log_mean(
        inner_diameter, inner_diameter + 2 * thickness_w
    )


def A_clean(inner_diameter, length_TLX, number_of_tubes_TLX):
    import numpy as np

    return inner_diameter * np.pi * length_TLX * number_of_tubes_TLX


def Ut_At(feed_types, flow, t_in, t_out, t_wall):
    return (
        heat_transferred(feed_types, flow, t_in, t_out)
        / log_mean(t_in - t_wall, t_out - t_wall)
        * 10 ** 6
    )


def TLE_efficiency(
    feed_types,
    bulk_temperatures,
    t_in,
    t_out,
    t_wall,
    flow,
    number_of_tubes_TLX,
    length_TLX,
    inner_diameter,
    outer_diameter,
    heat_transfer_coeff_adj_spyro,
):
    """
    Calculates the TLE efficiency taking into account lower flowrates and feed
    type
    """
    UtAt = Ut_At(feed_types, flow, t_in, t_out, t_wall)
    UcleanAclean = (
        A_clean(inner_diameter, length_TLX, number_of_tubes_TLX)
        * 1
        / U_clean_reciproce(
            feed_types,
            bulk_temperatures,
            t_in,
            t_out,
            t_wall,
            flow,
            number_of_tubes_TLX,
            inner_diameter,
            outer_diameter,
            heat_transfer_coeff_adj_spyro,
        )
    )

    return UtAt / UcleanAclean * 100


def Nusselt(
    feed_types, temperatures, flow, number_of_tubes_TLX, inner_diameter
):
    Re = Reynolds(
        feed_types, temperatures, flow, number_of_tubes_TLX, inner_diameter
    )
    Pr = Prandtl(feed_types, temperatures)

    return 0.023 * Re ** (0.8) * Pr ** (0.33)


def Reynolds(
    feed_types, temperatures, flow, number_of_tubes_TLX, inner_diameter
):
    import numpy as np

    mu = gass_visc(feed_types, temperatures)
    velocity_density = (
        flow / 3.6 / number_of_tubes_TLX / (inner_diameter ** 2 * np.pi / 4)
    )

    return velocity_density * inner_diameter / mu


def Prandtl(feed_types, temperatures):
    cp_gas = cp_gass(feed_types, temperatures)
    thermal_cond = gass_thermal_cond(feed_types, temperatures)
    viscosity = gass_visc(feed_types, temperatures)

    return cp_gas / thermal_cond * viscosity


def coke_thickness():
    #     (C7-(S7/PI()/AI7))/2*1000
    #     C7 internal diameter
    #     AI7 flow corrected = alpha_i
    #     S7 heat transfer coeff simplified
    pass
