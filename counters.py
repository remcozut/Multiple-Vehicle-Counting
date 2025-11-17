
class LineCrossCounter:

    def __init__(self, line_pos, line_orientation='vertical', direction='positive'):
        self.line_pos = line_pos
        self.line_orientation = line_orientation
        self.direction = direction
        self.counted_ids = {}
        self.counted = 0

    def count(self, bbox, object_id) -> bool:
        is_counted = False

        left, top, right, bottom = bbox

        if self.line_orientation == 'vertical':
            c = (left + right) // 2
        else:
            c = (top + bottom) // 2


        if self.direction == 'positive':  # left_to_right
            if c < self.line_pos:
                self.counted_ids[object_id] = c
        else:  # negative  # right_to_left
            if c > self.line_pos:
                self.counted_ids[object_id] = c

        offset = 0  # tolerance to avoid multiple counts due to detection noise
        if object_id in self.counted_ids:
            if self.direction == 'positive':  # left_to_right
                if self.counted_ids[object_id] + offset <= self.line_pos < c - offset:
                    self.counted = self.counted + 1
                    is_counted = True
            else:  # negative  # right_to_left
                if self.counted_ids[object_id] - offset  >= self.line_pos > c + offset:
                    self.counted = self.counted + 1
                    is_counted = True
            self.counted_ids[object_id] = c
        return is_counted

    def get_count(self):
        return self.counted


