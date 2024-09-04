import random

def cria(numVertices):
    file = open(f"NovasInstancias\Instancia_{numVertices}Vertices.txt", "w")
    file.write(str(numVertices))

    for i in range(numVertices):
        file.write("\n")
        for j in range(numVertices):
            elemento = str(random.randint(1, 20))
            file.write(elemento + " ")
    
    file.write("\n\n")

    for i in range(numVertices):
        elemento = str(random.randint(50, 200))
        file.write(elemento + " ")


cria(10)

