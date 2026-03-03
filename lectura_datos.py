def read_serie(name):
    with open(name, "r") as file:
        content = file.read()
        content = content.replace("[", "").replace("]", "")
        serie = [float(value) for value in content.split()]
    return serie

