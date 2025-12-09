import React from 'react';
import Layout from '@theme/Layout';

function ChatbotPage() {
  return (
    <Layout title="Chatbot" description="Physical AI & Humanoid Robotics Assistant">
      <div style={{ padding: '20px', maxWidth: '1000px', margin: '0 auto' }}>
        <div className="container">
          <div className="row">
            <div className="col col--12">
              <h1>Physical AI & Humanoid Robotics Assistant</h1>
              <p>The chatbot is now available as a floating widget in the bottom-right corner of every page.</p>
              <p>Simply click the 🤖 button to open the chat interface and ask questions about the book content.</p>
              <p>You can also select text on any page and use the "Ask about selection" feature to get explanations.</p>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default ChatbotPage;