import React, { useState } from 'react';

const ResizeNodePopup = ({ onConfirm, onCancel }) => {
  const [width, setWidth] = useState('');
  const [height, setHeight] = useState('');

  const handleConfirm = () => {
    // Validate width and height (add more validation as needed)
    if (!width || !height || isNaN(width) || isNaN(height)) {
      alert('Please enter valid width and height values.');
      return;
    }

    onConfirm(parseFloat(width), parseFloat(height));
  };

  return (
    <div className="resize-node-popup">
      <h3>Resize Node</h3>
      <label>
        Width:
        <input type="number" value={width} onChange={(e) => setWidth(e.target.value)} />
      </label>
      <label>
        Height:
        <input type="number" value={height} onChange={(e) => setHeight(e.target.value)} />
      </label>
      <button onClick={handleConfirm}>Confirm</button>
      <button onClick={onCancel}>Cancel</button>
    </div>
  );
};

export default ResizeNodePopup;
