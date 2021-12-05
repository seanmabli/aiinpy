import ReactDOM from 'react-dom';
import { BrowserRouter as Router, Switch, Route } from 'react-router-dom';
import About from './about';
import Function from './function';
import './index.css';

ReactDOM.render(
  <Router>
    <Switch>
      <Route exact path="/" component={About} />
      <Route exact path="/:version" component={About} />
      <Route exact path="/:version/:id" component={Function} />
    </Switch>
  </Router>,

document.getElementById('root'));