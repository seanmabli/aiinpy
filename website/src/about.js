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
    </div>
  );
}

export default About;