import { useParams } from "react-router";
import info from './version'
import Navbar from './navbar';
import './index.css';

function About() {
  const { version } = useParams();
  if (version !== undefined) {
    info[0].currentversion = version;
  }

  return (
    <div className="site">
      <div className="about">
        <Navbar />
      </div>
      <div className="function">
        <p className="p box">about</p>
        <p className="h1">aiinpy is an open source artificial intelligence package for the python programming language.  aiinpy can be used to build neural networks (nn), convolutional neural networks (cnn), recurrent neural networks (rnn), long term short term memory networks (lstm), and gated recurrent units (gru).  these networks can be trained with backpropagation as well as neuroevolution.</p>
      </div>
    </div>
  );
}

export default About;