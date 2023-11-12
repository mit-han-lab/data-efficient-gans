from PIL import Image
from object_segmentation.lang_sam import LangSAM

class ObjectSegmentation():
    def __init__(self, device='cpu'):
        self.model = LangSAM(device=device)
    
    def crop_image_to_square(self, image_pil: Image.Image, bounding_box) -> Image.Image:
        left, top, right, bottom = bounding_box

        original_width = right - left
        original_height = bottom - top

        if original_width > original_height:
            expand = (original_width - original_height) // 2
            top -= expand
            bottom += expand
        else:
            expand = (original_height - original_width) // 2
            left -= expand
            right += expand
        
        cropped_image = image_pil.crop((left, top, right, bottom))

        return cropped_image
    
    def segment_image(self, image_pil: Image.Image, text_prompt) -> Image.Image:
        masks, boxes, phrases, logits = self.model.predict(image_pil, text_prompt)
        
        box = boxes[0].tolist()
        cropped = self.crop_image_to_square(image_pil, box)
        
        return cropped
