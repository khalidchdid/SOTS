"""
This class manage cmp_pdf, extract signal name, pdf_to_img, extract state (equal_not equal), extract yellow, type (binaire ou analogique)
TO Do : extract for each unique signal, the page in which it appears
"""
import pymupdf


class Cpdf_cmp:
    def __init__(self, pdf_path):
        self._pdf_path = pdf_path
        self._doc = pymupdf.open(pdf_path)
        self._bool_equality_state = self._extract_equality_state()
        self._list_tuples_yellow_rect_time = self._extract_yellow_rect_time()

    def _extract_equality_state(self):
        """
        return: True if PDF contains (equal) else false
        """
        page = self._doc[0]
        text = page.get_text().encode("utf8") # get plain text (is in UTF-8)
        list_text = text
        str_equality_state = list_text[list_text.index(b'('):list_text.index(b')')+1]
        bool_equality_state = True if str_equality_state == b'(equal)' else False
        return bool_equality_state
    
    def _extract_yellow_rect_time(self):
        """
        The time is normlized and expressed in percentage 
        return : [(yellow_1_apparition, yellow_1_disparition, discrepancy_type), (yellow_2_apparition, yellow_2_disparition, discrepancy_type), ...] expressed in (%)
        if equal return : []
        """
        if self._bool_equality_state == True:
            return []
        page = self._doc[0]

        # --- Find FAULT y-position ---
        fault_y0 = None
        fault_x1 = None
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("text") == "FAULT":
                        fault_y0 = float(span["origin"][1])
                        fault_x1 = float(span["bbox"][2])
                        break
                if fault_y0 is not None:
                    break
            if fault_y0 is not None:
                break

        if fault_y0 is None:
            raise ValueError("FAULT separator not found on page; cannot split above/below.")

        paths = page.get_drawings()

        # --- Collect only yellow rectangles ---
        yellow_rects = []
        paths = page.get_drawings()
        for path in paths:
            color = path.get("color")
            if not color:
                continue
            r, g, b = color
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
            if r == 255 and g == 229 and b == 0:
                rect = path.get("rect")
                if rect is not None and (rect.y1  < page.rect.y1) and (rect.y1 > page.rect.y0):
                    yellow_rects.append(pymupdf.Rect(rect))

        # --- Sort yellow rectangles: above FAULT (by x0), then below FAULT (by x0) ---
        above = sorted(
            [r for r in yellow_rects if r.y1 < fault_y0],
            key=lambda r: r.x0,
        )
        below = sorted(
            [r for r in yellow_rects if r.y1 > fault_y0],
            key=lambda r: r.x0,
        )

        list_tuples_rects = []

        axis = min([p for p in paths] , key=lambda p: abs(p["rect"][0] - fault_x1))
        figure_start_px = axis["rect"][2]

        # figure_start_px = paths[0]["rect"].x0
        figure_end_px = paths[0]["rect"].x1
        time_length_px = figure_end_px - figure_start_px

        for rect in above:
            # the time in which the rect appears (max is used to unsure it's not negative i.e. lower than 0%)
            t_apparition = 100 * (rect.x0 - figure_start_px) / time_length_px
            # the time in which the rect disappears (min is used to unsure it's not upper than the 100% of the pdf )
            t_disparition = 100 * (rect.x1 - figure_start_px) / time_length_px
            list_tuples_rects.append([round(t_apparition,2), round(t_disparition,2), 0]) # 2 digit after '.'

        for rect in below:
            # the time in which the rect appears (max is used to unsure it's not negative i.e. lower than 0%)
            t_apparition = 100 * (rect.x0 - figure_start_px) / time_length_px
            # the time in which the rect disappears (min is used to unsure it's not upper than the 100% of the pdf )
            t_disparition = 100 * (rect.x1 - figure_start_px) / time_length_px
            list_tuples_rects.append([round(t_apparition,2), round(t_disparition,2), 1]) # 2 digit after '.'

        return list_tuples_rects

    
    def get_yellow_time(self):
        """
        return : yellow time (list of tuples containing percentage)
        """
        return self._list_tuples_yellow_rect_time
    def get_equality_state(self):
        """
        return: True if equal else false
        """
        return self._bool_equality_state