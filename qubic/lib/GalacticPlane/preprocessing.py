import numpy as np

def get_common_range(bounds: np.ndarray):

    # TODO: per la lettura del file e' 3  e 4
    return bounds[:, 0].max(), bounds[:, 1].min()


def get_tes_mask(tes_asic: np.ndarray):
    """
    sono marcati con bit 1 i TES validi e i TES non validi sono settati con bit 0
    """

    mask = 0
    # considero solo tes_asic con valid pari a 1
    # considero solo le prime due colonne TES e ASIC (non mi serve piu' valid)
    valid_tes_asic = tes_asic[tes_asic[: , 2] == 1][: , :2]

    for tes, asic in valid_tes_asic:

        if asic == 2:
            tes += 128

        # crea una maschera di bit dove 0 corrisponde al tes non valido
        # e 1 al tes valido. se faccio bin(mask) vedo una serie di 0 e 1
        # in corrispondenza di "indici" che sono i TES (in ordine crescente da dx a sx)
        mask |= (1 << int(tes))

    print(f"{mask = }")
    print(f"{bin(mask) = }")
    print(f"{mask.bit_length() = }")

    return mask

def check_tes_validity(tes, mask):

    return mask & (1 << tes)


def get_valid_tes(mask, n_tes: int):

    valid_tes = np.zeros(n_tes, dtype=np.bool_)

    for mask_bit in range(mask.bit_length()):
        valid_tes[mask_bit] = mask & (1 << mask_bit)

    return valid_tes