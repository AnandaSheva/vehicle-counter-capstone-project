const xywhToxyxy = (xywh) => {
    const x1 = xywh[0] - xywh[2] / 2
    const y1 = xywh[1] - xywh[3] / 2
    const x2 = xywh[0] + xywh[2] / 2
    const y2 = xywh[1] + xywh[3] / 2
    return [parseInt(x1), parseInt(y1), parseInt(x2), parseInt(y2)]
};

