import os
import re
constraints = []
CUR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def init_def_file(filename="templates.txt"):
    filename = os.path.join(CUR_DIR, filename)
    if constraints:
        return
    with open(filename, "r") as file:
        content = file.read()
    for section in content.split('"""ACT::'):
        if section.strip():
            header, body = section.split("\n", 1)
            name = header.strip()
            body = body.split('"""')
            body = body[0]
            if (name == "CONSTRAINTS"):
                cur_list = []
                within = False
                while (body.count("`")):
                    body = body.split("`", 1)
                    if (within):
                        cur_list.append(body[0])
                    body = body[1]
                    within = not within
                cur_list = cur_list[:-1]
                out = 'instr.add_constraints(\n"""\n'
                out += "\n".join(cur_list)
                out += '\n""")'
                constraints.append(out)
    num_sections = len(constraints)
    print("Initialized file")
    print(f"Found {num_sections} sections in file {filename}")
    for i in constraints:
        print(i)
        print("\n")


init_def_file("targets/AMX/def.txt")
