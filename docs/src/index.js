import ReactDOM from 'react-dom';
import { BrowserRouter as Router, Switch, Route } from 'react-router-dom';
import About from './about';
import Function from './function';
import './index.css';

window.currentversion = '0.0.17';
window.versions = ["0.0.17", "0.0.16", "0.0.13", "0.0.11"];

ReactDOM.render(
  <Router>
    <Switch>
      <Route exact path="/" component={About} />
      <Route exact path="/:version" component={About} />
      <Route exact path="/:version/:func" component={Function} />
    </Switch>
  </Router>,

document.getElementById('root'));
