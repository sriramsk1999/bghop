import json


class Obj2Text:
    def __init__(self, lib=None):
        if lib is None:
            self.lib = None
            self.lower_lib = None
        else:
            print("loading word mapping: ", lib)
            self.lib = json.load(open(lib))
            self.lower_lib = {k.lower(): v for k, v in self.lib.items()}

        self.template = "an image of a hand grasping a {}"

    def __call__(self, text):
        if isinstance(text, str):
            if len(text) == 0:
                return ""
            if text.startswith("an image of a hand grasping"):
                # print('emmm nested text???')
                return text
            if self.lib is None:
                return self.template.format(text)
            else:
                try:
                    text = self.lower_lib[text.lower()]
                except KeyError:
                    try:
                        text = self.lib[text]
                    except KeyError:
                        print(f"cannot find {text} in lib???")
                return self.template.format(text)
        elif isinstance(text, list):
            return [self(t) for t in text]
