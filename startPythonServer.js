import { spawn } from 'child_process';
import { createProxyMiddleware } from 'http-proxy-middleware';

export default function startPythonServer(app, config = {host: 'localhost', port: 6379}) {
	// Start the Python Flask server
	// Arguments are server start, reddist server host and port
	const pythonProcess = spawn('python3',
		['python/server.py', config.host, config.port.toString()],
		{
			stdio: 'pipe',
			cwd: 'submodules/python.server',
		}
);

	// Listen for Python server output
	pythonProcess.stdout.on('data', (data) => {
		console.log(`Python Server Output: ${data}`);
	});

	// Capture errors
	pythonProcess.stderr.on('data', (data) => {
		console.error(`Python Server Error: ${data}`);
	});

	// When the Python server exits
	pythonProcess.on('close', (code) => {
		console.log(`Python Server Process exited with code ${code}`);
		if (code !== 0) {
			console.error('Python Server crashed, restarting...');
            setTimeout(() => {
                startPythonServer(app);
            }, 5000); // Restart after 5 seconds
		}
	});

	process.on('exit', () => {
		pythonProcess.kill();
	});

	console.log('Python server started...');
	// redirect from local to python server
	app.use('/python', createProxyMiddleware({
		target: 'http://localhost:5000', // Python server
		changeOrigin: true,
		pathRewrite: {'^/python': ''}
	}));

}