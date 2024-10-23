import pandas as pd
from scipy.spatial import distance
from skin_detector import SkinDetector
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MakeupColorSelector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.skin_detector = SkinDetector(image_path)
        self.skintone_makeup_shades = pd.read_csv("datasets/skintone_makeup_shades.csv")
        self.makeup_product_shades = pd.read_csv("datasets/product_shades.csv")
        self.skin_color = self.detect_skin_color()
        self.filtered_shades = self.skintone_makeup_shades[self.skintone_makeup_shades['skintone_type'] == self.skin_color[1]]
        self.closest_shade = self.find_closest_skintone_shade()
        self.closest_makeup_shades = self.find_closest_makeup_shades()

    def detect_skin_color(self):
        skin_color = self.skin_detector.process_image(number_of_colors=3, has_thresholding=True)
        print(skin_color)
        return skin_color

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]

    def color_similarity(self, color1, color2):
        return distance.euclidean(color1, color2)

    def find_closest_skintone_shade(self):
        detected_color = self.hex_to_rgb(self.skin_color[0])
        self.filtered_shades['similarity'] = self.filtered_shades['shade'].apply(
            lambda x: self.color_similarity(detected_color, self.hex_to_rgb(x))
        )
        closest_shade = self.filtered_shades.loc[self.filtered_shades['similarity'].idxmin()]
        return closest_shade

    def find_closest_makeup_shades(self):
        closest_shade_hex = self.closest_shade['shade']
        closest_shade_rgb = self.hex_to_rgb(closest_shade_hex)
        self.makeup_product_shades['similarity'] = self.makeup_product_shades['hex'].apply(
            lambda x: self.color_similarity(closest_shade_rgb, self.hex_to_rgb(x))
        )
        closest_makeup_shades = self.makeup_product_shades.nsmallest(5, 'similarity')
        return closest_makeup_shades[['brand','product', 'hex', 'similarity']]

    def print_results(self):
        print()
        print()
        print()
        print("Closest Shade to Skin Color:", self.closest_shade['shade'])
        print()
        print("Top 5 Closest Makeup Shades:")
        print(self.closest_makeup_shades[['brand', 'hex', 'similarity']])

    def plot_color_circles_plotly(self):
        fig = make_subplots(rows=1, cols=1)
        
        # Get the top 5 closest makeup shades
        makeup_shades = self.closest_makeup_shades[['brand', 'hex', 'similarity']]

        # Set x-axis positions for circles
        positions = range(1, 6)

        for i, (_, row) in enumerate(makeup_shades.iterrows()):
            hex_color = f"#{row['hex']}"
            brand = row['brand']
            similarity = row['similarity']
            infos = f"color: {hex_color}, brand: {brand}, similarity: {similarity}"
            
            # Plot a circle with the hex color
            fig.add_trace(go.Scatter(
                x=[positions[i]],
                y=[0],
                marker=dict(color=hex_color, size=20),
                mode="markers",
                text=infos,
            ))

        # Update axis titles
        fig.update_xaxes(title_text='Top 5 Makeup Shades', row=1, col=1)

        # Update layout
        fig.update_layout(title='Top 5 Closest Makeup Shades', title_x=0.5, showlegend=False, template='plotly_white')

        # Display the plot
        fig.show()

# Example Usage
selector = MakeupColorSelector(image_path="test_img/alek.jpg")
#print(selector.find_closest_makeup_shades())
selector.print_results()
selector.plot_color_circles_plotly()

