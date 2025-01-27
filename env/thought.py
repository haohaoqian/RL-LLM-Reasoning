class LLM_thought():
    def __init__(self, tid, thought, parent_id=[], child_id =[], depth = -1) -> None:
        self.tid = tid
        self.thought = thought
        self.parent_id = parent_id
        self.child_id = child_id
        self.depth = depth

    def get_thought(self):
        return self.thought
    
    def get_tid(self):
        return self.tid
    
    def get_parent_id(self):
        return self.parent_id
    
    def get_child_id(self):
        return self.child_id
    
    def add_child(self, child_id):
        self.child_id.append(child_id)

    def get_depth(self):
        return self.depth
    
    def set_thought(self, thought):
        self.thought = thought