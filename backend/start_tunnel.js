const localtunnel = require('localtunnel');

(async () => {
  const tunnel = await localtunnel({ port: 3000, subdomain: 'vfitting123' });
  console.log(`Tunnel running at: ${tunnel.url}`);

  tunnel.on('close', () => {
    console.log('Tunnel closed');
    process.exit(0);
  });

  // Keep Node process alive
  setInterval(() => {}, 1000 * 60 * 60);
})();
