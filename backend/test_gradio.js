const { client } = require("@gradio/client");

async function testVTON() {
  try {
    console.log("Connecting to Gradio Space...");
    const app = await client("yisol/IDM-VTON");
    console.log("Connected!");
    const endpoints = await app.view_api();
    console.log(JSON.stringify(endpoints, null, 2));
  } catch (err) {
    console.error("Error:", err);
  }
}

testVTON();
