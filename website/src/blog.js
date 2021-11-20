import { Link } from 'react-router-dom';
import content from './content';
import './index.css';

function blog() {
  return(
    <div className="site">
      <div className="about">
        <p className="box">sean mabli</p>
        <h2>I am the creator of aiinpy, an open source python package used to create mechine learning models.</h2>
      </div>
      <div className="blog">
        {content.map((item) => {
          return (
            <div className="box">
              <Link to={item.url} className={'link'}>
                {item.title}
              </Link> <br/>
              <h1 className="lighter">{item.published}</h1>
              <h1>&nbsp;{item.tag}</h1>
              <h2>{item.discription}</h2>
            </div>
          ) 
        })}
      </div>
    </div>
  )
}

export default blog;