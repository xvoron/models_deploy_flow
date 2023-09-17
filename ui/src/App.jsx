import React, { useCallback, useEffect, useState } from 'react';
import ReactFlow, {
    useNodesState,
    useEdgesState,
    addEdge,
    MiniMap,
    Controls,
    Background,
    Panel,
} from 'reactflow';

import 'reactflow/dist/style.css';
import ResizeNodePopup from './ResizeNodePopup';
import axios from 'axios';

const port = 5123;

const initialNodes = [
    { id: '1', position: { x: 0, y: 0 }, data: { label: 'Image' } },
    { id: '2', position: { x: 0, y: 100 }, data: { label: 'Resize' } },
    { id: '3', position: { x: 0, y: 200 }, data: { label: 'Model' } },
    { id: '4', position: { x: 0, y: 300 }, data: { label: 'Output' } },
];

const initialEdges = [
    { id: 'e1-2', source: '1', target: '2' },
    { id: 'e2-3', source: '2', target: '3' },
    { id: 'e3-4', source: '3', target: '4' },
];

export default function App() {
    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
    const [showResizePopup, setShowResizePopup] = useState(false); // State to show/hide the resize popup
    const [resizingNodeId, setResizingNodeId] = useState(null); // State to store the node being resized
    const [resizeDimensions, setResizeDimensions] = useState({ width: 0, height: 0 });

    const onConnect = useCallback((params) => setEdges((eds) => addEdge(params, eds)), [setEdges]);
    const onClick = useCallback(() => {
        const id = (nodes.length + 1).toString();
        const newNode = {
            id,
            position: { x: Math.random() * window.innerWidth - 100, y: Math.random() * window.innerHeight },
            data: { label: `Node ${id}` },
        };
        setNodes((n) => n.concat(newNode));
    }, [setNodes]);

    const logNodeStates = () => {
        console.log("Nodes:", nodes);
    };

    const openResizePopup = (nodeId) => {
      setResizingNodeId(nodeId);
      setShowResizePopup(true);
    };

    const closeResizePopup = () => {
      setResizingNodeId(null);
      setShowResizePopup(false);
    };

    const confirmResize = (width, height) => {
        // Update the node with the new dimensions
        setNodes((prevNodes) =>
          prevNodes.map((node) =>
            node.id === resizingNodeId
              ? {
                  ...node,
                  style: {
                    ...node.style,
                    width: `${width}px`,
                    height: `${height}px`,
                  },
                }
              : node
          )
        );

        // Close the pop-up
        closeResizePopup();
      };

    const sendRequestToBuild = () => {
        const data = {
            nodes: nodes,
            edges: edges
        };
        console.log('Sending request to build', data);
        console.log('Port', port);
        axios.post(`http://localhost:${port}/build`, data)
            .then(res => {
                console.log('Request sent successfully', res.data);
            })
            .catch(err => {
                console.log('Error sending request', err);
            });
    };

    return (
        <div style={{ width: '100vw', height: '100vh' }}>
        <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            fitView
        >
        <Controls />
        <MiniMap />
        <Background variant="dots" gap={12} size={1} />
        <Panel>
            <button onClick={onClick}>Add New Node</button>
            <button onClick={sendRequestToBuild}>Build</button>
        </Panel>
        </ReactFlow>

        {/* Render the resize pop-up */}
        {showResizePopup && (
          <ResizeNodePopup
            onConfirm={(width, height) => confirmResize(width, height)}
            onCancel={closeResizePopup}
          />
        )}
        </div>
    );
}
