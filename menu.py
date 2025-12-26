class Menu:
    def __init__(self, items:list, actions:list):
        self.items=items
        self.actions=actions
    
    def select(self, user_choice:str):
        try:
            return self.actions[int(user_choice)]()
        except IndexError:
            return "ERR: ITEM DOES NOT HAVE ACTION"
        except:
            return "ERR: UNKNOWN"
        
    def display_menu(self, flags="", custom_phrase="", prependix="", appendix=""):
        menu_string=f""
        match flags:
            case "custom":
                if custom_phrase:
                    menu_string+=prependix
                    for i in range(len(self.items)):
                        menu_string+=custom_phrase
                    menu_string+=appendix
                    return menu_string
                else:
                    return "ERR: no custom string supplied"
            case "show actions":
                pass
            case _:
                for i in range(len(self.items)):
                    menu_string+=f"{i}. {self.items[i]}\n"
                menu_string+=f"Please enter in the number to select the associated menu item."
                return menu_string
        