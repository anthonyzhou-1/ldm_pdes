# manually labeled shapes

class TurbLabels():
    def __init__(self, split):
        self.split = split 
        # manually list all possible data directories
        if self.split == "train":
            self.fnames = ['step-higher', 'step-lower', 'corner', 'opp-corners-asym', 'neighbor-corners', 'corners', 'pillar', 'offset-pillar', 'double-pillar', 'opp-pillar', 'bar', 'double-bar', 'teeth', 'offset-teeth', 'elbow', 'wide-elbow', 'elbow-snug', 'open-elbow', 'donut', 'U', 'H', 'T', 'disjoint-T', 'plus', 'minus', 'square', 'square-offset', '2x2', '2x2-large', '3x3', '3x3-inv', 'cross-wide', 'cross-offset', 'platform', 'high-platform', 'altar' ]
        elif self.split == "valid":
            self.fnames = ['step-low', 'opp-corners-sym', 'wide-pillar', 'elbow-asym', 'square-large', 'cross', 'wide-teeth', 'step-high', 'offset-bar']

        self.n_samples = len(self.fnames) # set the length of the dataset to number of geometries. In reality, each geometry has multiple data samples.
    
    def get_label(self, inputs):
        if isinstance(inputs, int):
            fname = self.fnames[inputs]
        else:
            fname = inputs # if inputs is a string

        position_label2 = None
        position_label3 = None 
        position_label4 = None
        position_label5 = None
        elbow = False 
        donut = False 
        H = False
        altar = False
        cross = False 
        plus = False

        if fname == "step-higher":
            shape = 'step'
            qualitative_label = "higher step at the bottom half, spanning the width of the domain and nearly half the height of the domain"
            shape_label = "24 units wide and 11 units high"
            position_label = "at the bottom of the domain, starting from y = 0 to y = 10 and x = 0 to x = 23"

        elif fname == "step-lower":
            shape = 'step'
            qualitative_label = "lower step at the bottom half, spanning the width of the domain and nearly a quarter the height of the domain"
            shape_label = "24 units wide and 5 units high"
            position_label = "at the bottom of the domain, starting from y = 0 to y = 4 and x = 0 to x = 23"

        elif fname == "corner":
            shape = 'square'
            qualitative_label = "medium-large square at the bottom left corner, spanning over a third the width of the domain and over a third the height of the domain"
            shape_label = "9 units wide and 9 units high"
            position_label = "at the bottom left of the domain, starting from y = 0 to y = 8 and x = 0 to x = 8"
        
        elif fname == "opp-corners-asym":
            shape = 'square'
            qualitative_label = "two asymmetric squares at opposite corners. The square at the bottom left is small and the square at the top right is medium in size"
            shape_label = "6 units wide and 6 units high"
            position_label = "at the bottom left of the domain, starting from y = 0 to y = 5 and x = 0 to x = 5"
            shape_label2 = "8 units wide and 8 units high"
            position_label2 = "at the top right of the domain, starting from y = 16 to y = 23 and x = 16 to x = 23"

        elif fname == "neighbor-corners":
            shape = 'square'
            qualitative_label = "two medium squares at neighboring corners. The square at the bottom left is nearly equal in size to the square at the top left"
            shape_label = "8 units wide and 8 units high"
            position_label = "at the bottom left of the domain, starting from y = 0 to y = 7 and x = 0 to x = 7"
            shape_label2 = "8 units wide and 7 units high"
            position_label2 = "at the top left of the domain, starting from y = 17 to y = 23 and x = 0 to x = 7"
        
        elif fname == "corners":
            shape = 'square'
            qualitative_label = "four small squares at all four corners. The squares, which are located at the bottom left, top left, bottom right, and top right are all equal in size"
            shape_label = "6 units wide and 6 units high"
            position_label = "at the bottom left of the domain, starting from y = 0 to y = 5 and x = 0 to x = 5"
            shape_label2 = "6 units wide and 6 units high"
            position_label2 = "at the top left of the domain, starting from y = 18 to y = 23 and x = 0 to x = 5"
            shape_label3 = "6 units wide and 6 units high"
            position_label3 = "at the bottom right of the domain, starting from y = 0 to y = 5 and x = 18 to x = 23"
            shape_label4 = "6 units wide and 6 units high"
            position_label4 = "at the top right of the domain, starting from y = 18 to y = 23 and x = 18 to x = 23"
        
        elif fname == "pillar":
            shape = 'pillar'
            qualitative_label = "narrower, vertical pillar at the bottom center of the domain"
            shape_label = "4 units wide and 16 units high"
            position_label = "at the bottom center of the domain, starting from y = 0 to y = 15 and x = 10 to x = 13"
        
        elif fname == "offset-pillar":
            shape = 'pillar'
            qualitative_label = "offset narrow, vertical pillar at the bottom center-left of the domain"
            shape_label = "5 units wide and 16 units high"
            position_label = "at the bottom center of the domain, starting from y = 0 to y = 15 and x = 5 to x = 9"
        
        elif fname == "double-pillar":
            shape = 'pillar'
            qualitative_label = "two narrower, vertical pillars. The pillar on the left is equal in size to the pillar on the right"
            shape_label = "4 units wide and 16 units high"
            position_label = "at the bottom center-left of the domain, starting from y = 0 to y = 15 and x = 5 to x = 8"
            shape_label2 = "4 units wide and 16 units high"
            position_label2 = "at the bottom center-right of the domain, starting from y = 0 to y = 15 and x = 15 to x = 18"
        
        elif fname == "opp-pillar":
            shape = 'pillar'
            qualitative_label = "two narrow, vertical pillars that are opposite of each other. The pillar on the bottom left is equal in size to the pillar on the top right"
            shape_label = "5 units wide and 16 units high"
            position_label = "at the bottom center-left of the domain, starting from y = 0 to y = 15 and x = 5 to x = 9"
            shape_label2 = "5 units wide and 16 units high"
            position_label2 = "at the top center-right of the domain, starting from y = 8 to y = 23 and x = 15 to x = 19"
        
        elif fname == "bar":
            shape = 'bar'
            qualitative_label = "vertical bar at the center of the domain, spanning the entire height of the domain"
            shape_label = "6 units wide and 24 units high"
            position_label = "at the bottom center of the domain, starting from y = 0 to y = 23 and x = 9 to x = 14"
        
        elif fname == "double-bar":
            shape = 'bar'
            qualitative_label = "two narrow, vertical bars that span the entire height of the domain. The two bars are equal in size to each other"
            shape_label = "5 units wide and 24 units high"
            position_label = "at the bottom center-left of the domain, starting from y = 0 to y = 23 and x = 5 to x = 9"
            shape_label2 = "5 units wide and 24 units high"
            position_label2 = "at the bottom center-right of the domain, starting from y = 0 to y = 23 and x = 15 to x = 19"
        
        elif fname == "teeth":
            shape = "rectangle"
            qualitative_label = "two rectangles that each span over a third of the height of the domain. The two rectangles are nearly equal in size to each other and are vertically opposite"
            shape_label = "6 units wide and 8 units high"
            position_label = "at the bottom center of the domain, starting from y = 0 to y = 7 and x = 9 to x = 14"
            shape_label2 = "6 units wide and 10 units high"
            position_label2 = "at the top center of the domain, starting from y = 14 to y = 23 and x = 9 to x = 14"
        
        elif fname == "offset-teeth":
            shape = "rectangle"
            qualitative_label = "two rectangles that each span over a third of the height of the domain. The two rectangles are nearly equal in size to each other and are opposite of each other"
            shape_label = "6 units wide and 8 units high"
            position_label = "at the bottom center-left of the domain, starting from y = 0 to y = 7 and x = 5 to x = 10"
            shape_label2 = "6 units wide and 10 units high"
            position_label2 = "at the top center-right of the domain, starting from y = 14 to y = 23 and x = 11 to x = 16"
        
        elif fname == "elbow":
            shape = "elbow"
            qualitative_label = "an elbow that consists of two vertical and horizontal pillars that are joined together"
            shape_label = "4 units wide and 10 units high"
            position_label = "at the bottom center of the domain, starting from y = 0 to y = 9 and x = 10 to x = 13"
            shape_label2 = "14 units wide and 4 units high"
            position_label2 = "at the center right of the domain, starting from y = 10 to y = 13 and x = 10 to x = 23"
            elbow = True
        
        elif fname == "wide-elbow":
            shape = "elbow"
            qualitative_label = "a thick elbow that consists of two thick vertical and horizontal pillars that are joined together"
            shape_label = "7 units wide and 10 units high"
            position_label = "at the bottom center of the domain, starting from y = 0 to y = 9 and x = 10 to x = 16"
            shape_label2 = "14 units wide and 7 units high"
            position_label2 = "at the center right of the domain, starting from y = 10 to y = 16 and x = 10 to x = 23"
            elbow = True 
        
        elif fname == "elbow-snug":
            shape = "elbow"
            qualitative_label = "a long and thick elbow that consists of two thick vertical and long horizontal pillars that are joined together. The horizontal pillar spans the width of the domain"
            shape_label = "8 units wide and 10 units high"
            position_label = "at the bottom left of the domain, starting from y = 0 to y = 9 and x = 0 to x = 7"
            shape_label2 = "24 units wide and 6 units high"
            position_label2 = "at the center of the domain, starting from y = 11 to y = 16 and x = 0 to x = 23"
            elbow = True 
        
        elif fname == "open-elbow":
            shape = "elbow"
            qualitative_label = "an open elbow that consists of two small vertical and a small horizontal pillars that are not joined together with a hole in the middle"
            shape_label = "5 units wide and 8 units high"
            position_label = "at the bottom center-left of the domain, starting from y = 0 to y = 7 and x = 8 to x = 12"
            shape_label2 = "8 units wide and 5 units high"
            position_label2 = "at the center right of the domain, starting from y = 12 to y = 16 and x = 16 to x = 23"
            elbow = True 
        
        elif fname == "donut": 
            shape = 'square'
            qualitative_label = "donut that consists of a larger square at the center of the domain with a smaller square hole in the center"
            shape_label = "11 units wide and 11 units high"
            position_label = "at the center of the domain, starting from y = 7 to y = 17 and x = 7 to x = 17"
            shape_label2 = "5 units wide and 5 units high"
            position_label2 = "at the center of the domain, starting from y = 10 to y = 14 and x = 10 to x = 14"
            donut = True 
        
        elif fname == "U":
            shape = 'square'
            qualitative_label = "U shape that consists of a larger square at the center of the domain with a rectangle hole in the top center"
            shape_label = "11 units wide and 11 units high"
            position_label = "at the center of the domain, starting from y = 7 to y = 17 and x = 7 to x = 17"
            shape_label2 = "5 units wide and 8 units high"
            position_label2 = "at the top center of the domain, starting from y = 10 to y = 17 and x = 10 to x = 14"
            donut = True

        elif fname == "H":
            shape = 'square'
            qualitative_label = "H shape that consists of a larger square  at the center of the domain with two rectangle holes. The first hole is at the bottom center and the second hole is at the top center"
            shape_label = "11 units wide and 11 units high"
            position_label = "at the center of the domain, starting from y = 7 to y = 17 and x = 7 to x = 17"
            shape_label2 = "5 units wide and 4 units high"
            position_label2 = "at the bottom center of the domain, starting from y = 7 to y = 10 and x = 10 to x = 14"
            shape_label3 = "5 units wide and 4 units high"
            position_label3 = "at the top center of the domain, starting from y = 14 to y = 17 and x = 10 to x = 14"
            H = True
        
        elif fname == "T":
            shape = "T shape"
            qualitative_label = "a T shape that consists of two vertical and a horizontal pillars that are joined together at the center of the domain"
            shape_label = "5 units wide and 10 units high"
            position_label = "at the center of the domain, starting from y = 5 to y = 14 and x = 10 to x = 14"
            shape_label2 = "15 units wide and 4 units high"
            position_label2 = "at the top center of the domain, starting from y = 15 to y = 18 and x = 5 to x = 19"
            elbow = True

        elif fname == "disjoint-T":
            shape = "T shape"
            qualitative_label = "an open T shape that consists of two small vertical and a horizontal pillars that are not joined together with a hole in the middle"
            shape_label = "5 units wide and 7 units high"
            position_label = "at the center of the domain, starting from y = 5 to y = 11 and x = 10 to x = 14"
            shape_label2 = "15 units wide and 4 units high"
            position_label2 = "at the top center of the domain, starting from y = 15 to y = 18 and x = 5 to x = 19"
            elbow = True
        
        elif fname == "plus":
            shape = 'horizontal pillar'
            qualitative_label = "plus shape that consists of a horizontal pillar that is joined together with two squares at the bottom center and top center"
            shape_label = "14 units wide and 5 units high"
            position_label = "at the center of the domain, starting from y = 10 to y = 14 and x = 5 to x = 18"
            shape_label2 = "5 units wide and 5 units high"
            position_label2 = "at the bottom center of the domain, starting from y = 5 to y = 9 and x = 10 to x = 14"
            shape_label3 = "5 units wide and 4 units high"
            position_label3 = "at the top center of the domain, starting from y = 15 to y = 18 and x = 10 to x = 14"
            plus = True 
        
        elif fname == "minus":
            shape = 'pillar'
            qualitative_label = "horizontal pillar at the bottom center, near the middle of the domain"
            shape_label = "14 units wide and 5 units high"
            position_label = "at the bottom center of the domain, starting from y = 7 to y = 11 and x = 5 to x = 18"
        
        elif fname == "square":
            shape = 'square'
            qualitative_label = "medium square at the center of the domain"
            shape_label = "8 units wide and 8 units high"
            position_label = "at the center of the domain, starting from y = 8 to y = 15 and x = 8 to x = 15"
        
        elif fname == "square-offset":
            shape = 'square'
            qualitative_label = "large square offset from the center of the domain, shifted to the bottom center and center left"
            shape_label = "10 units wide and 10 units high"
            position_label = "at the bottom center and center left of the domain, starting from y = 5 to y = 14 and x = 5 to x = 14"
        
        elif fname == "2x2":
            shape = "square"
            qualitative_label = "two small squares that are near the center of the domain. The two squares are equal in size"
            shape_label = "6 units wide and 6 units high"
            position_label = "at the bottom center-left of the domain, starting from y = 6 to y = 11 and x = 6 to x = 11"
            shape_label2 = "6 units wide and 6 units high"
            position_label2 = "at the top center-right of the domain, starting from y = 12 to y = 17 and x = 12 to x = 17"
        
        elif fname == "2x2-large":
            shape = "square"
            qualitative_label = "two medium squares that are near the center of the domain. The two squares are equal in size"
            shape_label = "8 units wide and 8 units high"
            position_label = "at the bottom center-left of the domain, starting from y = 4 to y = 11 and x = 4 to x = 11"
            shape_label2 = "8 units wide and 8 units high"
            position_label2 = "at the top center-right of the domain, starting from y = 12 to y = 20 and x = 12 to x = 20"
        
        elif fname == "3x3":
            shape = "square"
            qualitative_label = "five smaller squares. All five squares are equal in size"
            shape_label = "5 units wide and 5 units high"
            position_label = "at the bottom center-left of the domain, starting from y = 5 to y = 9 and x = 5 to x = 9"
            position_label2 = "at the bottom center-right of the domain, starting from y = 5 to y = 9 and x = 15 to x = 19"
            position_label3 = "at the center of the domain, starting from y = 10 to y = 14 and x = 10 to x = 14"
            position_label4 = "at the top center-left of the domain, starting from y = 15 to y = 19 and x = 5 to x = 9"
            position_label5 = "at the top center-right of the domain, starting from y = 15 to y = 19 and x = 15 to x = 19"
        
        elif fname == "3x3-inv":
            shape = "square"
            qualitative_label = "four smaller squares. All four squares are equal in size"
            shape_label = "5 units wide and 5 units high"
            position_label = "at the bottom center of the domain, starting from y = 5 to y = 9 and x = 10 to x = 14"
            shape_label2 = "5 units wide and 5 units high"
            position_label2 = "at the center left of the domain, starting from y = 10 to y = 14 and x = 5 to x = 9"
            shape_label3 = "5 units wide and 5 units high"
            position_label3 = "at the center right of the domain, starting from y = 10 to y = 14 and x = 15 to x = 19"
            shape_label4 = "5 units wide and 5 units high"
            position_label4 = "at the top center of the domain, starting from y = 15 to y = 19 and x = 10 to x = 14"
        
        elif fname == "cross-wide":
            shape = 'bar'
            qualitative_label = "wide cross shape that consists of a narrow vertical bar that is joined together with two thick horizontal pillars at the center left and center right"
            shape_label = "5 units wide and 24 units high"
            position_label = "at the bottom center of the domain, starting from y = 0 to y = 23 and x = 10 to x = 14"
            shape_label2 = "10 units wide and 8 units high"
            position_label2 = "at the center left of the domain, starting from y = 6 to y = 13 and x = 0 to x = 9"
            shape_label3 = "9 units wide and 8 units high"
            position_label3 = "at the center right of the domain, starting from y = 6 to y = 13 and x = 15 to x = 23"
            cross = True 
        
        elif fname == "cross-offset":
            shape = 'bar'
            qualitative_label = "offset cross shape that consists of a narrow vertical bar that is joined together with two narrow horizontal pillars at the top-center left and top-center right"
            shape_label = "5 units wide and 24 units high"
            position_label = "at the bottom center of the domain, starting from y = 0 to y = 23 and x = 7 to x = 11"
            shape_label2 = "7 units wide and 5 units high"
            position_label2 = "at the top-center left of the domain, starting from y = 14 to y = 18 and x = 0 to x = 6"
            shape_label3 = "12 units wide and 5 units high"
            position_label3 = "at the top-center right of the domain, starting from y = 14 to y = 18 and x = 12 to x = 23"
            cross = True 
        
        elif fname == "platform":
            shape = 'horizontal pillar'
            qualitative_label = "narrow horizontal pillar at the bottom center of the domain"
            shape_label = "18 units wide and 5 units high"
            position_label = "at the bottom center of the domain, starting from y = 0 to y = 4 and x = 3 to x = 20"
        
        elif fname == "high-platform":
            shape = 'horizontal pillar'
            qualitative_label = "thick horizontal pillar at the bottom center of the domain"
            shape_label = "14 units wide and 9 units high"
            position_label = "at the bottom center of the domain, starting from y = 0 to y = 8 and x = 5 to x = 18"
        
        elif fname == "altar":
            shape = "square"
            qualitative_label = "an altar that consists of a thick horizontal pillar at the bottom center joined together with a medium rectangle at the center of the domain"
            shape_label = "14 units wide and 7 units high"
            position_label = "at the bottom center of the domain, starting from y = 0 to y = 6 and x = 5 to x = 18"
            shape_label2 = "6 units wide and 7 units high"
            position_label2 = "at the center of the domain, starting from y = 7 to y = 13 and x = 9 to x = 14"
            altar = True
        
        if fname == "step-low":
            shape = 'step'
            qualitative_label = "low step at the bottom half, spanning the width of the domain and over a quarter the height of the domain"
            shape_label = "24 units wide and 7 units high"
            position_label = "at the bottom of the domain, starting from y = 0 to y = 6 and x = 0 to x = 23"
        
        elif fname == "opp-corners-sym":
            shape = 'square'
            qualitative_label = "two medium squares at opposite corners. The two squares are nearly equal in size"
            shape_label = "8 units wide and 8 units high"
            position_label = "at the bottom left of the domain, starting from y = 0 to y = 7 and x = 0 to x = 7"
            shape_label2 = "7 units wide and 7 units high"
            position_label2 = "at the top right of the domain, starting from y = 17 to y = 23 and x = 17 to x = 23"
        
        elif fname == "wide-pillar":
            shape = 'pillar'
            qualitative_label = "thick, vertical pillar at the bottom center of the domain"
            shape_label = "8 units wide and 16 units high"
            position_label = "at the bottom center of the domain, starting from y = 0 to y = 15 and x = 8 to x = 15"

        elif fname == "elbow-asym":
            shape = "elbow"
            qualitative_label = "a thick elbow that consists of two thick vertical and horizontal pillars that are joined together"
            shape_label = "8 units wide and 10 units high"
            position_label = "at the bottom center of the domain, starting from y = 0 to y = 9 and x = 10 to x = 16"
            shape_label2 = "14 units wide and 6 units high"
            position_label2 = "at the center right of the domain, starting from y = 10 to y = 15 and x = 10 to x = 23"
            elbow = True 
        
        elif fname == "square-large":
            shape = 'square'
            qualitative_label = "larger square at the center of the domain"
            shape_label = "11 units wide and 11 units high"
            position_label = "at the center of the domain, starting from y = 7 to y = 17 and x = 7 to x = 17"
        
        elif fname == "cross":
            shape = 'bar'
            qualitative_label = "narrow cross shape that consists of a narrow vertical bar that is joined together with two narrow horizontal pillars at the center left and center right"
            shape_label = "5 units wide and 24 units high"
            position_label = "at the bottom center of the domain, starting from y = 0 to y = 23 and x = 10 to x = 14"
            shape_label2 = "10 units wide and 5 units high"
            position_label2 = "at the center left of the domain, starting from y = 10 to y = 14 and x = 0 to x = 9"
            shape_label3 = "9 units wide and 5 units high"
            position_label3 = "at the center right of the domain, starting from y = 10 to y = 14 and x = 15 to x = 23"
            cross = True 
        
        elif fname == "wide-teeth":
            shape = "rectangle"
            qualitative_label = "two wide rectangles that each span over a third of the height of the domain. The two rectangles are nearly equal in size to each other and are vertically opposite"
            shape_label = "10 units wide and 8 units high"
            position_label = "at the bottom center of the domain, starting from y = 0 to y = 7 and x = 7 to x = 16"
            shape_label2 = "10 units wide and 10 units high"
            position_label2 = "at the top center of the domain, starting from y = 14 to y = 23 and x = 7 to x = 16"
        
        elif fname == "step-high":
            shape = 'step'
            qualitative_label = "high step at the bottom half, spanning the width of the domain and over a third the height of the domain"
            shape_label = "24 units wide and 9 units high"
            position_label = "at the bottom of the domain, starting from y = 0 to y = 8 and x = 0 to x = 23"
        
        elif fname == "offset-bar":
            shape = 'bar'
            qualitative_label = "offset vertical bar at the center right of the domain, spanning the entire height of the domain"
            shape_label = "6 units wide and 24 units high"
            position_label = "at the center right of the domain, starting from y = 0 to y = 23 and x = 14 to x = 19"

        if altar:
            return f"Air flows over an obstacle resembling {qualitative_label}. The first {shape} is {shape_label} and is located at {position_label}. The second square is {shape_label2} and is located at {position_label2}. The flow is turbulent."

        if cross:
            return f"Air flows over an obstacle resembling a {qualitative_label}. The {shape} is {shape_label} and is located at {position_label}. The first pillar is {shape_label2} and is located at {position_label2}. The second pillar is {shape_label3} and is located at {position_label3}. The flow is turbulent."

        if plus:
            return f"Air flows over an obstacle resembling a {qualitative_label}. The {shape} is {shape_label} and is located at {position_label}. The first square is {shape_label2} and is located at {position_label2}. The second square is {shape_label3} and is located at {position_label3}. The flow is turbulent."
        
        if H:
            return f"Air flows over an obstacle resembling a {qualitative_label}. The {shape} is {shape_label} and is located at {position_label}. The first hole is {shape_label2} and is located at {position_label2}. The second hole is {shape_label3} and is located at {position_label3}. The flow is turbulent."

        if donut:
            return f"Air flows over an obstacle resembling a {qualitative_label}. The {shape} is {shape_label} and is located at {position_label}. The hole is {shape_label2} and is located at {position_label2}. The flow is turbulent."

        if elbow:
            return f"Air flows over an obstacle resembling {qualitative_label}. The first vertical pillar is {shape_label} and is located at {position_label}. The second horizontal pillar is {shape_label2} and is located at {position_label2}. The flow is turbulent."
        
        if position_label5 is not None:
            return f"Air flows over an obstacle resembling {qualitative_label}. The first {shape} is {shape_label} and is located at {position_label}. The second {shape} is {shape_label} and is located at {position_label2}. The third {shape} is {shape_label} and is located at {position_label3}. The fourth {shape} is {shape_label} and is located at {position_label4}. The fifth {shape} is {shape_label} and is located at {position_label5}. The flow is turbulent."
        elif position_label4 is not None:
            return f"Air flows over an obstacle resembling {qualitative_label}. The first {shape} is {shape_label} and is located at {position_label}. The second {shape} is {shape_label2} and is located at {position_label2}. The third {shape} is {shape_label3} and is located at {position_label3}. The fourth {shape} is {shape_label4} and is located at {position_label4}. The flow is turbulent."
        elif position_label2 is not None:
            return f"Air flows over an obstacle resembling {qualitative_label}. The first {shape} is {shape_label} and is located at {position_label}. The second {shape} is {shape_label2} and is located at {position_label2}. The flow is turbulent."
        else:
            return f"Air flows over an obstacle resembling a {qualitative_label}. The {shape} is {shape_label} and is located at {position_label}. The flow is turbulent."
