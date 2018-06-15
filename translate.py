import csv
import io

def _foo():
    with io.open("./jappo.csv", 'r', encoding="utf-8") as csvfile:

        recipes = csv.reader(csvfile, delimiter=';')
        f = open("./jappo_translate.txt", "w")
        for recipe in recipes:
            for ingredient in recipe:
                f.write(ingredient.lower() + "\n")
            f.write("####\n")

        f.close()


if __name__ == '__main__':

    with io.open("./jappo_translated.rtf", "r") as f:

        line = f.readlines()[9:-1]
        res = ",".join(line).replace("\\\n", "").replace(',####,', '\n')

        s = io.open("./jappo_final.csv", "w")
        s.write(res)
