import Navbar from './navbar';
import { useParams } from "react-router";
import './index.css';

function About() {
  const { version } = useParams();

  console.log('about.js');
  if (window.versions.includes(version)) {
    window.currentversion = version;
  }

  return (
      <div>
        <p className="p box">about</p>
        <p className="h1 box">aiinpy is an open source artificial intelligence package for the python programming language.  aiinpy can be used to build neural networks (nn), convolutional neural networks (cnn), recurrent neural networks (rnn), long term short term memory networks (lstm), and gated recurrent units (gru).  these networks can be trained with backpropagation as well as neuroevolution.</p>
        <div className="box">
          <p className="h1">install aiinpy through pypi:&nbsp;</p>
          <p className="h1 bold">pip install aiinpy</p>
        </div>
        <div className="box">
          <p className="h1">contribute to aiinpy on github:&nbsp;</p>
          <a href="https://github.com/seanmabli/aiinpy" className="h1 bold link">seanmabli/aiinpy</a> 
        </div>
      </div>
  );
}

export default About;