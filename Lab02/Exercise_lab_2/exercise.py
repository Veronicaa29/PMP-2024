import random

def este_prim(numar):
    if numar < 2:
        return False
    for i in range(2, numar):
        if numar % i == 0:
            return False
    return True

def adauga_bila(urna):
    numar = arunca_zarul()
    # print("zar: ",  numar)
    if este_prim(numar):
        urna[2] += 1
    elif numar == 6:
        urna[0] += 1
    else:
        urna[1] += 1

    return urna

def extragere(urna):
    bile = []
    for i in range(len(urna)):
        for j in range(urna[i]):
            bile.append(i)
    bila = random.randint(0, len(bile) - 1) # indexul bilei alese
    return bile[bila]

def arunca_zarul():
    return random.randint(1, 6)

def probabilitate_bila_rosie(urna, simulari = 3000):
    bile_rosii = 0
    urna_initiala = urna.copy()
    for _ in range(simulari):
        urna = urna_initiala.copy()
        adauga_bila(urna)
        bila_extrasa = extragere(urna)
        if bila_extrasa == 0:
            bile_rosii += 1
    return bile_rosii / simulari

def main():
    urna = [3, 4, 2] # rosu, albastru, negru
    print(probabilitate_bila_rosie(urna))
    adauga_bila(urna)
    bila = extragere(urna)
    print(bila) # 0 - rosu, 1 - albastru, 2 - negru

if __name__ == "__main__":
    main()